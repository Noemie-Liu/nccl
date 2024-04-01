/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

#define LOC_HOSTMEM 0
#define LOC_DEVMEM  1
#define LOC_COUNT   2

struct netSendResources {
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  char* buffers[LOC_COUNT];
  int buffSizes[LOC_COUNT];
  void* mhandles[LOC_COUNT];
  void** mhandlesProto[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  char* buffers[LOC_COUNT];
  int buffSizes[LOC_COUNT];
  void* mhandles[LOC_COUNT];
  void** mhandlesProto[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

/* Determine if two peers can communicate with NET */
ncclResult_t netCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  NCCLCHECK(ncclTopoGetNetDev(topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(topo, myInfo->busId, resources->netDev, 1, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->recvMem, 1));

  send->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;
  send->conn.tail = &resources->recvMem->tail;
  send->conn.fifo = resources->recvMem->sizesFifo;
  send->conn.head = &resources->sendMem->head;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.fifo[i] = -1;

  int protoLoc[NCCL_NUM_PROTOCOLS];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    protoLoc[p] = p != NCCL_PROTO_LL && resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
  }

  int buffSizes[NCCL_NUM_PROTOCOLS];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // Only allocate buffers for simple for p2p connections
    buffSizes[p] = graph == NULL && p != NCCL_PROTO_SIMPLE ? 0 : send->comm->buffSizes[p];
    resources->buffSizes[protoLoc[p]] += buffSizes[p];
  }

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclCudaCalloc(resources->buffers+LOC_DEVMEM, resources->buffSizes[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclCudaHostCalloc(resources->buffers+LOC_HOSTMEM, resources->buffSizes[LOC_HOSTMEM]));
  }

  int offsets[LOC_COUNT];
  offsets[LOC_HOSTMEM] = offsets[LOC_DEVMEM] = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->mhandlesProto[p] = resources->mhandles+protoLoc[p];
    send->conn.buffs[p] = resources->buffers[protoLoc[p]] + offsets[protoLoc[p]];
    offsets[protoLoc[p]] += buffSizes[p];
  }

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

ncclResult_t netRecvSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId) {
  struct netRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  NCCLCHECK(ncclTopoGetNetDev(topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(topo, myInfo->busId, resources->netDev, 0, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->recvMem, 1));

  recv->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;
  recv->conn.tail = &resources->recvMem->tail;
  recv->conn.head = &resources->sendMem->head;

  int protoLoc[NCCL_NUM_PROTOCOLS];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    protoLoc[p] = resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
  }

  int buffSizes[NCCL_NUM_PROTOCOLS];
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    // Only allocate buffers for simple for p2p connections
    buffSizes[p] = graph == NULL && p != NCCL_PROTO_SIMPLE ? 0 : recv->comm->buffSizes[p];
    resources->buffSizes[protoLoc[p]] += buffSizes[p];
  }

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclCudaCalloc(resources->buffers+LOC_DEVMEM, resources->buffSizes[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclCudaHostCalloc(resources->buffers+LOC_HOSTMEM, resources->buffSizes[LOC_HOSTMEM]));
  }

  int offsets[LOC_COUNT];
  offsets[LOC_HOSTMEM] = offsets[LOC_DEVMEM] = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->mhandlesProto[p] = resources->mhandles+protoLoc[p];
    recv->conn.buffs[p] = resources->buffers[protoLoc[p]] + offsets[protoLoc[p]];
    offsets[protoLoc[p]] += buffSizes[p];
  }

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [receive] via NET/%s/%d%s", channelId, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetListen(resources->netDev, &info->netHandle, &resources->netListenComm));

  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;

  // Connect to remote peer
  NCCLCHECK(ncclNetConnect(resources->netDev, info->netHandle, &resources->netSendComm));

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[LOC_DEVMEM], resources->buffSizes[LOC_DEVMEM], NCCL_PTR_CUDA, &resources->mhandles[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[LOC_HOSTMEM], resources->buffSizes[LOC_HOSTMEM], NCCL_PTR_HOST, &resources->mhandles[LOC_HOSTMEM]));
  }
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t netRecvConnect(struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct netRecvResources* resources = (struct netRecvResources*)recv->transportResources;

  // Finish connection establishment from remote peer
  NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[LOC_DEVMEM], resources->buffSizes[LOC_DEVMEM], NCCL_PTR_CUDA, &resources->mhandles[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[LOC_HOSTMEM], resources->buffSizes[LOC_HOSTMEM], NCCL_PTR_HOST, &resources->mhandles[LOC_HOSTMEM]));
  }
  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  for (int l=0; l<LOC_COUNT; l++) {
    if (resources->buffers[l])
      NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->mhandles[l]));
  }
  NCCLCHECK(ncclCudaHostFree(resources->buffers[LOC_HOSTMEM]));
  CUDACHECK(cudaFree(resources->buffers[LOC_DEVMEM]));
  NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  for (int l=0; l<LOC_COUNT; l++) {
    if (resources->buffers[l])
      NCCLCHECK(ncclNetDeregMr(resources->netRecvComm, resources->mhandles[l]));
  }
  NCCLCHECK(ncclCudaHostFree(resources->buffers[LOC_HOSTMEM]));
  CUDACHECK(cudaFree(resources->buffers[LOC_DEVMEM]));
  NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
  free(resources);
  return ncclSuccess;
}

#define cut_size 10240
ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct netSendResources* resources = (struct netSendResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = args->connector->comm->buffSizes[p] / NCCL_STEPS;
    char* localBuff = args->connector->conn.buffs[p];
    void* mhandle = *(resources->mhandlesProto[p]);
    args->idle = 1;
    int total_num_cut = 0;
    if (args->head < args->end) {
      int buffSlot = args->tail%NCCL_STEPS;
      volatile int* sizesFifo = resources->recvMem->sizesFifo;
      int num_cut = (sizesFifo[buffSlot]-1)/cut_size + 1;
      total_num_cut += num_cut;
      if (args->tail < args->end && args->tail < args->head + NCCL_STEPS) {
        //volatile int* sizesFifo = resources->recvMem->sizesFifo;
        //int num_cut = (sizesFifo[buffSlot]-1)/cut_size + 1;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        if (args->protocol == NCCL_PROTO_LL128) {
          if (args->tail < *recvTail) {
            if (sizesFifo[buffSlot] != -1) {
              int ready = resources->useGdr;
              if (!ready) {
                // When data is in sysmem, we need to wait until all flags are correct since the GPU only
                // called threadfence()
                uint64_t flag = args->tail + 1;
                int nFifoLines = DIVUP(sizesFifo[buffSlot], sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
                volatile uint64_t* lines = (volatile uint64_t*)(localBuff+buffSlot*stepSize);
                ready = 1;
                for (int i=0; i<nFifoLines; i++) {
                  if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
                }
              }
              if (ready) {
                // Send through network
                int test = 0;
                for (int i=0; i<num_cut; i++){
                  int real_cut_size = i==num_cut-1? sizesFifo[buffSlot]-i*cut_size : cut_size;
                  NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize+i*cut_size, real_cut_size, mhandle, args->requests+total_num_cut-num_cut+i));
                  if (args->requests[total_num_cut-num_cut+i] == NULL){
                    test = 1;
                  }
                }
                if (test == 0) {
                  sizesFifo[buffSlot] = -1;
                  // Make sure size is reset to zero before we update the head.
                  __sync_synchronize();
                  args->tail += args->sliceSteps;
                  args->idle = 0;
                }
              }
            }
          }
        } else if (args->protocol == NCCL_PROTO_LL) {
          int size = sizesFifo[buffSlot];
          for (int i=0; i<num_cut; i++){
            int real_cut_size = i==num_cut-1? sizesFifo[buffSlot]-i*cut_size : cut_size;
            if (real_cut_size != -1) {
              uint32_t flag = NCCL_LL_FLAG(args->tail + 1);
              int nFifoLines = DIVUP(real_cut_size, sizeof(union ncclLLFifoLine));
              union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize+i*real_cut_size);
              real_cut_size = nFifoLines * sizeof(union ncclLLFifoLine);
              int ready = 1;
              for (int j=0; j<nFifoLines; j++) {
                volatile uint32_t *f1 = &lines[j].flag1;
                volatile uint32_t *f2 = &lines[j].flag2;
                if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
              }
              if (ready){
                NCCLCHECK(ncclNetIsend(resources->netSendComm, lines, real_cut_size, mhandle, args->requests+total_num_cut-num_cut+i));
              }
            }
          }
          int test = 0;
          for (int i=0; i<num_cut; i++){
            if (args->requests[total_num_cut-num_cut+i] == NULL){
              test = 1;
            }
          }
          if (test == 0) {
            sizesFifo[buffSlot] = -1;
            // Make sure size is reset to zero before we update the head.
            __sync_synchronize();
            args->tail += args->sliceSteps;
            args->idle = 0;
          }
        } else if (args->tail < *recvTail) {
          int test = 0;
          // Send through network
          if (sizesFifo[buffSlot] != -1) {
            for (int i=0; i<num_cut; i++){
                int real_cut_size = i==num_cut-1? sizesFifo[buffSlot]-i*cut_size : cut_size;
                NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize+i*cut_size, real_cut_size, mhandle, args->requests+total_num_cut-num_cut+i));
                if (args->requests[total_num_cut-num_cut+i] == NULL){
                    test = 1;
                }
            }
            if (test == 0) {
              sizesFifo[buffSlot] = -1;
              // Make sure size is reset to zero before we update the head.
              __sync_synchronize();
              args->tail += args->sliceSteps;
              args->idle = 0;
            }
          }
        }
      }
      if (args->head < args->tail) {
        int done;
        int buffSlot = args->head%NCCL_STEPS;
        for (int i=0; i<num_cut; i++){
          NCCLCHECK(ncclNetTest(args->requests[total_num_cut-num_cut+i], &done, NULL));
        }
        if (done) {
          args->head += args->sliceSteps;
          resources->sendMem->head = args->head;
          args->idle = 0;
        }
      }
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct netRecvResources* resources = (struct netRecvResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    args->idle = 1;
    int p = args->protocol;
    int stepSize = args->connector->comm->buffSizes[p] / NCCL_STEPS;
    char* localBuff = args->connector->conn.buffs[p];
    void* mhandle = *(resources->mhandlesProto[p]);
    int total_num_cut = 0;
    if (args->head < args->end) {
      volatile uint64_t* sendHead = &resources->sendMem->head;
      //int total_num_cut = 0;
      if ((args->tail < args->head + NCCL_STEPS) && (args->tail < *sendHead + NCCL_STEPS) && (args->tail < args->end)) {
        int buffSlot = args->tail%NCCL_STEPS;
        int sliceSize = stepSize * args->sliceSteps;
        int num_cut = (sliceSize-1)/cut_size+1;
        int test = 0;
        for (int i=0; i<num_cut; i++){
          int real_cut_size = i==num_cut-1? sliceSize-i*cut_size : cut_size;
          NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+buffSlot*stepSize+i*cut_size, real_cut_size, mhandle, args->requests+total_num_cut+i));
          if (args->requests[total_num_cut+i] == NULL) {
            step = 1;
          }
        }
        args->tail += args->sliceSteps;
        args->idle = 0;
      }
      if (args->tail > args->head) {
        int buffSlot = args->head%NCCL_STEPS;
        //int done, size;
        int sliceSize = stepSize * args->sliceSteps;
        int num_cut = (sliceSize-1)/cut_size+1;
        for (int i=0; i<num_cut; i++){
          int done, size;
          NCCLCHECK(ncclNetTest(args->requests[total_num_cut+i], &done, &size));
          if (done){
            if (args->protocol == NCCL_PROTO_SIMPLE){
              if (resources->useGdr) NCCLCHECK(ncclNetFlush(resources->netRecvComm, localBuff+buffSlot*stepSize+i*cut_size, size, mhandle));
            }
          }
        }
        args->head += args->sliceSteps;
        if (args->protocol == NCCL_PROTO_SIMPLE) resources->recvMem->tail = args->head;
          args->idle = 0;
      }
      //int sliceSize = stepSize * args->sliceSteps;
      //int num_cut = (sliceSize-1)/cut_size+1;
      total_num_cut += num_cut;
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

/*
//新版本，扰乱流的顺序
#define cut_size 10240
ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct netSendResources* resources = (struct netSendResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = args->connector->comm->buffSizes[p] / NCCL_STEPS;
    char* localBuff = args->connector->conn.buffs[p];
    void* mhandle = *(resources->mhandlesProto[p]);
    args->idle = 1;
    int total_num_cut = 0;
    if (args->head < args->end) {
      int buffSlot = args->tail%NCCL_STEPS;
      volatile int* sizesFifo = resources->recvMem->sizesFifo;
      int num_cut = (sizesFifo[buffSlot]-1)/cut_size + 1;
      total_num_cut += num_cut;
      if (args->tail < args->end && args->tail < args->head + NCCL_STEPS) {
        //volatile int* sizesFifo = resources->recvMem->sizesFifo;
        //int num_cut = (sizesFifo[buffSlot]-1)/cut_size + 1;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        if (args->protocol == NCCL_PROTO_LL128) {
          if (args->tail < *recvTail) {
            if (sizesFifo[buffSlot] != -1) {
              int ready = resources->useGdr;
              if (!ready) {
                // When data is in sysmem, we need to wait until all flags are correct since the GPU only
                // called threadfence()
                uint64_t flag = args->tail + 1;
                int nFifoLines = DIVUP(sizesFifo[buffSlot], sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
                volatile uint64_t* lines = (volatile uint64_t*)(localBuff+buffSlot*stepSize);
                ready = 1;
                for (int i=0; i<nFifoLines; i++) {
                  if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
                }
              }
              if (ready) {
                // Send through network
                int test = 0;
                for (int i=0; i<num_cut; i++){
                  int real_cut_size = i==num_cut-1? sizesFifo[buffSlot]-i*cut_size : cut_size;
                  if (i==num_cut-2) continue;
                  NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize+i*cut_size, real_cut_size, mhandle, args->requests+total_num_cut-num_cut+i));
                  if (args->requests[total_num_cut-num_cut+i] == NULL){
                    test = 1;
                  }
                }
                if (num_cut >== 2){
                  NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize+(num_cut-2)*cut_size, cut_size, mhandle, args->requests+total_num_cut-2));
                  if (args->requests[total_num_cut-2] == NULL){
                    test = 1;
                  }
                }
                if (test == 0) {
                  sizesFifo[buffSlot] = -1;
                  // Make sure size is reset to zero before we update the head.
                  __sync_synchronize();
                  args->tail += args->sliceSteps;
                  args->idle = 0;
                }
              }
            }
          }
        } else if (args->protocol == NCCL_PROTO_LL) {
          int size = sizesFifo[buffSlot];
          for (int i=0; i<num_cut; i++){
            int real_cut_size = i==num_cut-1? sizesFifo[buffSlot]-i*cut_size : cut_size;
            if (real_cut_size != -1) {
              uint32_t flag = NCCL_LL_FLAG(args->tail + 1);
              int nFifoLines = DIVUP(real_cut_size, sizeof(union ncclLLFifoLine));
              union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize+i*real_cut_size);
              real_cut_size = nFifoLines * sizeof(union ncclLLFifoLine);
              int ready = 1;
              for (int j=0; j<nFifoLines; j++) {
                volatile uint32_t *f1 = &lines[j].flag1;
                volatile uint32_t *f2 = &lines[j].flag2;
                if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
              }
              if (ready){
                if (i==num_cut-2) continue;
                NCCLCHECK(ncclNetIsend(resources->netSendComm, lines, real_cut_size, mhandle, args->requests+total_num_cut-num_cut+i));
              }
            }
          }
          if (num_cut >=2){
            int nFifoLines = DIVUP(cut_size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize+(num_cut-2)*cut_size);
            real_cut_size = nFifoLines * sizeof(union ncclLLFifoLine);
            NCCLCHECK(ncclNetIsend(resources->netSendComm, lines, real_cut_size, mhandle, args->requests+total_num_cut-2));
          }

          int test = 0;
          for (int i=0; i<num_cut; i++){
            if (args->requests[total_num_cut-num_cut+i] == NULL){
              test = 1;
            }
          }
          if (test == 0) {
            sizesFifo[buffSlot] = -1;
            // Make sure size is reset to zero before we update the head.
            __sync_synchronize();
            args->tail += args->sliceSteps;
            args->idle = 0;
          }
        } else if (args->tail < *recvTail) {
          int test = 0;
          // Send through network
          if (sizesFifo[buffSlot] != -1) {
            for (int i=0; i<num_cut; i++){
                int real_cut_size = i==num_cut-1? sizesFifo[buffSlot]-i*cut_size : cut_size;
                if (i==num_cut-2) continue;
                NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize+i*cut_size, real_cut_size, mhandle, args->requests+total_num_cut-num_cut+i));
                if (args->requests[total_num_cut-num_cut+i] == NULL){
                    test = 1;
                }
            }
            if (num_cut >= 2){
              NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize+(num_cut-2)*cut_size, cut_size, mhandle, args->requests+total_num_cut-2));
              if (args->requests[total_num_cut-2] == NULL){
                test = 1;
              }
            }

            if (test == 0) {
              sizesFifo[buffSlot] = -1;
              // Make sure size is reset to zero before we update the head.
              __sync_synchronize();
              args->tail += args->sliceSteps;
              args->idle = 0;
            }
          }
        }
      }
      if (args->head < args->tail) {
        int done;
        int buffSlot = args->head%NCCL_STEPS;
        for (int i=0; i<num_cut; i++){
          NCCLCHECK(ncclNetTest(args->requests[total_num_cut-num_cut+i], &done, NULL));
        }
        if (done) {
          args->head += args->sliceSteps;
          resources->sendMem->head = args->head;
          args->idle = 0;
        }
      }
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct netRecvResources* resources = (struct netRecvResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    args->idle = 1;
    int p = args->protocol;
    int stepSize = args->connector->comm->buffSizes[p] / NCCL_STEPS;
    char* localBuff = args->connector->conn.buffs[p];
    void* mhandle = *(resources->mhandlesProto[p]);
    int total_num_cut = 0;
    if (args->head < args->end) {
      volatile uint64_t* sendHead = &resources->sendMem->head;
      //int total_num_cut = 0;
      if ((args->tail < args->head + NCCL_STEPS) && (args->tail < *sendHead + NCCL_STEPS) && (args->tail < args->end)) {
        int buffSlot = args->tail%NCCL_STEPS;
        int sliceSize = stepSize * args->sliceSteps;
        int num_cut = (sliceSize-1)/cut_size+1;
        int test = 0;
        for (int i=0; i<num_cut; i++){
          int real_cut_size = i==num_cut-1? sliceSize-i*cut_size : cut_size;
          if (i==num_cut-2) continue;
          NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+buffSlot*stepSize+i*cut_size, real_cut_size, mhandle, args->requests+total_num_cut+i));
          if (args->requests[total_num_cut+i] == NULL) {
            step = 1;
          }
        }
        if (num_cut >=2){
          NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+buffSlot*stepSize+(num_cut-2)*cut_size, cut_size, mhandle, args->requests+total_num_cut+num_cut-2));
          if (args->requests[total_num_cut+num_cut-2] == NULL) {
            step = 1;
          }
        }
        args->tail += args->sliceSteps;
        args->idle = 0;
      }
      if (args->tail > args->head) {
        int buffSlot = args->head%NCCL_STEPS;
        //int done, size;
        int sliceSize = stepSize * args->sliceSteps;
        int num_cut = (sliceSize-1)/cut_size+1;
        for (int i=0; i<num_cut; i++){
          if (i==num_cut-2) break;
          int done, size;
          NCCLCHECK(ncclNetTest(args->requests[total_num_cut+i], &done, &size));
          if (done){
            if (args->protocol == NCCL_PROTO_SIMPLE){
              if (resources->useGdr) NCCLCHECK(ncclNetFlush(resources->netRecvComm, localBuff+buffSlot*stepSize+i*cut_size, size, mhandle));
            }
          }
        }
        if (num_cut >= 2){
          int done, size;
          NCCLCHECK(ncclNetTest(args->requests[total_num_cut+num_cut-2], &done, &size));
          if (done){
            if (args->protocol == NCCL_PROTO_SIMPLE){
              if (resources->useGdr) NCCLCHECK(ncclNetFlush(resources->netRecvComm, localBuff+buffSlot*stepSize+(num_cut-2)*cut_size, size, mhandle));
            }
          }
        }
        args->head += args->sliceSteps;
        if (args->protocol == NCCL_PROTO_SIMPLE) resources->recvMem->tail = args->head;
          args->idle = 0;
      }
      //int sliceSize = stepSize * args->sliceSteps;
      //int num_cut = (sliceSize-1)/cut_size+1;
      total_num_cut += num_cut;
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}
*/
struct ncclTransport netTransport = {
  "NET",
  netCanConnect,
  { netSendSetup, netSendConnect, netSendFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxy }
};
