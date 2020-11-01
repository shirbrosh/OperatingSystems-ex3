#include "MapReduceFramework.h"
#include "Barrier.h"
#include <pthread.h>
#include <vector>
#include <atomic>
#include <iostream>
#include <deque>

#define MAP_PHASE_LOCK_FAILURE "failed to lock in map phase"
#define MAP_PHASE_UNLOCK_FAILURE "failed to unlock in map phase"
#define SHUFFLE_LOCK_FAILURE "shuffle lock mutex failed"
#define SHUFFLE_UNLOCK_FAILURE "shuffle unlock mutex failed"
#define REDUCE_LOCK_FAILURE "reduce lock mutex failed"
#define REDUCE_UNLOCK_FAILURE "reduce unlock mutex failed"
#define LOCK_CHANGING_STATE_FAILURE "failed changing state in shuffle"
#define UNLOCK_CHANGING_STATE_FAILURE "failed changing state in shuffle unlock"
#define PTHREAD_JOIN_FAILED "pthread_join failed"
#define LOCK_THREAD_MUTEX_FAILURE_EIMT2 "emit2 lock mutex failed"
#define UNLOCK_THREAD_MUTEX_FAILURE_EIMT2 "emit2 unlock mutex failed"
#define LOCK_THREAD_MUTEX_FAILURE "lock thread mutex failed"
#define UNLOCK_THREAD_MUTEX_FAILURE "unlock thread mutex failed"
#define LOCK_FAILURE_EIMT3 "emit3 lock mutex failed"
#define UNLOCK_FAILURE_EIMT3 "emit3 unlock mutex failed"
#define SHUFFLE_THREAD 0
#define SYS_ERR "system error: "
#define PTHREAD_CREATE_FAIL "failed to create thread"

struct ThreadContext;
struct Job;

/**
 * An expansion for JobHandle typedef
 */
struct Job
{
    JobState state = {UNDEFINED_STAGE, 0};
    int inputSize;

    std::atomic<int> atomicCounterPairs;
    std::atomic<int> atomicCounterMapPhase;
    std::atomic<int> atomicCounterMapsDone;
    std::atomic<int> atomicCounterShuffleDone;
    std::atomic<int> atomicCounterReducePhase;
    std::atomic<int> atomicCounterReduceDone;
    std::atomic<int> atomicCounterThreadsReadyToShuffle;

    Barrier *barrier;
    const MapReduceClient *client;
    const InputVec *inputVec;
    OutputVec *outputVec;
    std::vector<pthread_t> threads;
    std::deque<int> threadsReadyToShuffle;
    std::vector<ThreadContext> threadContextVec;
    IntermediateMap shuffleRes;

    pthread_mutex_t emit3Mutex;
    pthread_mutex_t emit2Mutex;
    pthread_mutex_t reduceMutex;
    pthread_mutex_t stateMutex;
    pthread_mutex_t mapMutex;
    pthread_mutex_t shuffleMutex;
};

/**
 * This function initialize the fields of the Job -initialJob
 * @param initialJob - the Job to initialize
 * @param client - The implementation of MapReduceClient class, in other words the task that the
 *          framework should run.
 * @param inputVe -  a vector of type std::vector<std::pair<K1*, V1*>>, the input elements
 * @param outputVec-  a vector of type std::vector<std::pair<K3*, V3*>>, to which the output
 *          elements will be added before returning. You can assume that outputVec is empty.
 * @param multiThreadLevel - the number of worker threads to be used for running the algorithm.
 *
 */
void initJob(Job *initialJob, const MapReduceClient *client, const InputVec *inputVec,
			 OutputVec *outputVec, int multiThreadLevel)
{
    initialJob->state = {MAP_STAGE, 0};
    initialJob->inputSize = (int)inputVec->size();

    std::atomic_init(&initialJob->atomicCounterPairs, 0);
    std::atomic_init(&initialJob->atomicCounterMapPhase, 0);
    std::atomic_init(&initialJob->atomicCounterMapsDone, 0);
    std::atomic_init(&initialJob->atomicCounterShuffleDone, 0);
    std::atomic_init(&initialJob->atomicCounterReducePhase, 0);
    std::atomic_init(&initialJob->atomicCounterReduceDone, 0);
    std::atomic_init(&initialJob->atomicCounterThreadsReadyToShuffle, 0);

    initialJob->barrier= new Barrier(multiThreadLevel);
    initialJob->client = client;
    initialJob->inputVec = inputVec;
    initialJob->outputVec = outputVec;
    initialJob->threadContextVec = std::vector<ThreadContext>(multiThreadLevel);
    initialJob->threads = std::vector<pthread_t>(multiThreadLevel);

    initialJob->emit3Mutex = pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
    initialJob->emit2Mutex = pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
    initialJob->reduceMutex = pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
    initialJob->stateMutex = pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
    initialJob->mapMutex = pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
    initialJob->shuffleMutex = pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
}

/**
 * A wrapper for Job struct containing the ID of a specific thread
 */
struct ThreadContext
{
    int ThreadID;
    Job *job;
    pthread_mutex_t threadMutex;
    std::vector<IntermediatePair> emit2output;
};

/**
 * lock the given mutex and print the massage if it fail.
 * @param mutex mutex to lock
 * @param errMsg massage to print
 */
void lockMutex(pthread_mutex_t *mutex, const std::string &errMsg)
{
	if (pthread_mutex_lock(mutex))
	{
		std::cerr << SYS_ERR << errMsg << std::endl;
		exit(1);
	}
}

/**
 * unlock the given mutex and print the massage if it fail.
 * @param mutex mutex to lock
 * @param errMsg massage to print
 */
void unlockMutex(pthread_mutex_t *mutex, const std::string &errMsg)
{
	if (pthread_mutex_unlock(mutex))
	{
		std::cout << SYS_ERR << errMsg << std::endl;
		exit(1);
	}
}


/**
 * This function reads pairs of (k1,v1) from the input vector and calls the map function on
 * each of them
 * @param threadContext - the context of the thread preforming this phase
 */
void mapPhase(ThreadContext *threadContext)
{
    int oldVal = threadContext->job->atomicCounterMapPhase++;
    while (oldVal < threadContext->job->inputSize)
    {
        auto inputPair = (*threadContext->job->inputVec)[oldVal];
        threadContext->job->client->map(inputPair.first, inputPair.second, threadContext);

        lockMutex(&threadContext->job->shuffleMutex, MAP_PHASE_LOCK_FAILURE);
        threadContext->job->threadsReadyToShuffle.push_back(threadContext->ThreadID);
        threadContext->job->atomicCounterThreadsReadyToShuffle++;
        unlockMutex(&threadContext->job->shuffleMutex, MAP_PHASE_UNLOCK_FAILURE);

        threadContext->job->atomicCounterMapsDone++;
        oldVal = threadContext->job->atomicCounterMapPhase++;
    }
}


/**
 * This function reads the Map phase output and combines them into a single IntermediateMap (of type
 * std::map< k2*, std::vector<v2*> >)
 * @param threadContext- the context of the thread preforming this phase
 */
void shufflePhase(ThreadContext *threadContext)
{
    while (threadContext->job->atomicCounterMapsDone<threadContext->job->inputSize ||
           threadContext->job->atomicCounterThreadsReadyToShuffle>0)
    {
        //in case there are no threads that finished the map phase and ready to shuffle
        if (threadContext->job->atomicCounterThreadsReadyToShuffle==0)
        {
            continue;
        }

        //find the ID of thread that finished the map phase and ready to shuffle
        lockMutex(&threadContext->job->shuffleMutex, SHUFFLE_LOCK_FAILURE);
        int threadToShuffle = threadContext->job->threadsReadyToShuffle.front();
        threadContext->job->threadsReadyToShuffle.pop_front();
        threadContext->job->atomicCounterThreadsReadyToShuffle--;
        unlockMutex(&threadContext->job->shuffleMutex, SHUFFLE_UNLOCK_FAILURE);

        //perform the shuffle function
        lockMutex(&threadContext->job->threadContextVec[threadToShuffle].threadMutex,
                LOCK_THREAD_MUTEX_FAILURE);
        for (auto &it : threadContext->job->threadContextVec[threadToShuffle].emit2output)
        {
            threadContext->job->shuffleRes[it.first].push_back(it.second);
            threadContext->job->atomicCounterShuffleDone++;
        }
        threadContext->job->threadContextVec[threadToShuffle].emit2output.clear();
        unlockMutex(&threadContext->job->threadContextVec[threadToShuffle].threadMutex,
                    UNLOCK_THREAD_MUTEX_FAILURE);

        //chang the job stage from map to shuffle
        if (threadContext->job->atomicCounterMapsDone==threadContext->job->inputSize)
        {
            lockMutex(&threadContext->job->stateMutex, LOCK_CHANGING_STATE_FAILURE);
            threadContext->job->state.stage = SHUFFLE_STAGE;
            unlockMutex(&threadContext->job->stateMutex, UNLOCK_CHANGING_STATE_FAILURE);
        }
    }
}

/**
 * In this function each thread reads a key k2 and calls the reduce function using the key and it’s
 * value from the IntermediateMap. The reduce function in turn will call the emit3 function to
 * output (k3,v3) pairs which inserted directly to the output vector.
 * @param threadContext- the context of the thread preforming this phase
 */
void reducePhase(ThreadContext *threadContext)
{
    lockMutex(&threadContext->job->stateMutex, LOCK_CHANGING_STATE_FAILURE);
    if (threadContext->job->state.stage != REDUCE_STAGE)
    {
        threadContext->job->state.stage = REDUCE_STAGE;
        threadContext->job->state.percentage = 0;
    }
    while (threadContext->job->atomicCounterReduceDone != (int)threadContext->job->shuffleRes
            .size())
    {
        lockMutex(&threadContext->job->reduceMutex,REDUCE_LOCK_FAILURE);
        auto it = threadContext->job->shuffleRes.begin();
        int oldVal = threadContext->job->atomicCounterReducePhase++;
        for (int i = 0; i < oldVal; i++)
        {
            ++it;
        }
        K2 *key = (*it).first;
        threadContext->job->client->reduce(key, threadContext->job->shuffleRes[key], threadContext);
        threadContext->job->atomicCounterReduceDone++;
        unlockMutex(&threadContext->job->reduceMutex,REDUCE_UNLOCK_FAILURE);
    }
    unlockMutex(&threadContext->job->stateMutex, UNLOCK_CHANGING_STATE_FAILURE);
}

/**
 * The function that is given to a thread during it's creation
 * @param ThreadContext- the context of the thread preforming this phase
 * @return The thread context
 */
void *jobExecute(void *threadContext)
{
    auto curThreadContext = (ThreadContext *) threadContext;
    if (curThreadContext->ThreadID == SHUFFLE_THREAD)
    {
        shufflePhase(curThreadContext);
    }
    else
    {

        mapPhase(curThreadContext);
    }
    curThreadContext->job->barrier->barrier();
    reducePhase(curThreadContext);
    return threadContext;
}


/**
 * This function starts running the MapReduce algorithm (with several threads) and returns a
 * JobHandle
* @param client - The implementation of MapReduceClient class, in other words the task that the
 *          framework should run.
 * @param inputVe - a vector of type std::vector<std::pair<K1*, V1*>>, the input elements
 * @param outputVec-  a vector of type std::vector<std::pair<K3*, V3*>>, to which the output
 *          elements will be added before returning. You can assume that outputVec is empty.
 *
 * @param multiThreadLevel - the number of worker threads to be used for running the algorithm.
 * @return JobHandle object
 */
JobHandle startMapReduceJob(const MapReduceClient &client, const InputVec &inputVec, OutputVec
&outputVec, int multiThreadLevel)
{
    Job *initialJob = new Job;
    initJob(initialJob, &client, &inputVec, &outputVec, multiThreadLevel);
    for (int i = 0; i < multiThreadLevel; i++)
    {
        initialJob->threadContextVec[i] =  ThreadContext{i, initialJob, pthread_mutex_t
                (PTHREAD_MUTEX_INITIALIZER)};
        if (pthread_create(&initialJob->threads[i], nullptr, jobExecute, &
                (initialJob->threadContextVec[i])))
        {
            std::cerr << SYS_ERR << PTHREAD_CREATE_FAIL << std::endl;
            exit(1);
        }
    }
    return initialJob;
}


/**
 * This function gets the job handle returned by startMapReduceFramework and waits until it is
 * finished.
 * @param job- Job struct object
 */
void waitForJob(JobHandle job)
{
    auto waitJob = (Job *) job;
    for (auto it = waitJob->threads.begin(); it != waitJob->threads.end(); ++it)
    {
        if (pthread_join((*it), nullptr))
        {
            std::cerr << SYS_ERR << PTHREAD_JOIN_FAILED << std::endl;
            exit(1);
        }
    }
}

/**
 * this function gets a job handle and updates the state of the job into the given JobState struct.
 * @param job- Job struct object
 * @param state- JobState typedef object
 */
void getJobState(JobHandle job, JobState *state)
{
    auto stateJob = (Job *) job;
    int keyAmountInMap;
    lockMutex(&stateJob->stateMutex, LOCK_CHANGING_STATE_FAILURE);
    switch (stateJob->state.stage)
    {
        case MAP_STAGE:
            stateJob->state.percentage = ((float) stateJob->atomicCounterMapsDone * 100)
                                         / (float) stateJob->inputSize;
            break;
        case SHUFFLE_STAGE:
            stateJob->state.percentage = ((float) stateJob->atomicCounterShuffleDone * 100) /
                                         (float) stateJob->atomicCounterPairs;
            break;
        case REDUCE_STAGE:
            keyAmountInMap = (int)stateJob->shuffleRes.size();
            stateJob->state.percentage = ((float)stateJob->atomicCounterReduceDone *100) /
                                         (float) keyAmountInMap;
    }
    state->stage = stateJob->state.stage;
    state->percentage = stateJob->state.percentage;
    unlockMutex(&stateJob->stateMutex, UNLOCK_CHANGING_STATE_FAILURE);
}

/**
 * This function produces a (K2*,V2*) pair.
 * @param key - a K2 pointer
 * @param value-  a V2 pointer
 * @param threadContext - The context can be used to get pointers into the framework’s variables and data
 * structures. Its exact type is implementation dependent
 */
void emit2(K2 *key, V2 *value, void *threadContext)
{
    auto getContext = (ThreadContext *) threadContext;
    lockMutex(&getContext->threadMutex, LOCK_THREAD_MUTEX_FAILURE_EIMT2);
    getContext->emit2output.emplace_back(IntermediatePair{key, value});
    unlockMutex(&getContext->threadMutex, UNLOCK_THREAD_MUTEX_FAILURE_EIMT2);
    getContext->job->atomicCounterPairs++;
}

/**
 * This function produces a (K3*,V3*) pair. It has the following signature
 * @param key - a K2 pointer
 * @param value-  a V2 pointer
 * @param threadContext - The context can be used to get pointers into the framework’s variables and data
 * structures. Its exact type is implementation dependent
 */
void emit3(K3 *key, V3 *value, void *threadContext)
{
    auto getContext = (ThreadContext *) threadContext;
    lockMutex(&getContext->job->emit3Mutex, LOCK_FAILURE_EIMT3);
    getContext->job->outputVec->push_back(OutputPair{key, value});
    unlockMutex(&getContext->job->emit3Mutex, UNLOCK_FAILURE_EIMT3);
}

/**
 * Free all memory resources of the given job
 * @param job - job to free.
 */
void closeJobHandle(JobHandle job)
{
    auto closeJob = (Job *) job;
    waitForJob(job);
    for(auto context: closeJob->threadContextVec){
        context.job= nullptr;
        pthread_mutex_destroy(&context.threadMutex);
    }
    pthread_mutex_destroy(&closeJob->emit2Mutex);
    pthread_mutex_destroy(&closeJob->reduceMutex);
    pthread_mutex_destroy(&closeJob->stateMutex);
    pthread_mutex_destroy(&closeJob->mapMutex);
    pthread_mutex_destroy(&closeJob->shuffleMutex);
    pthread_mutex_destroy(&closeJob->emit3Mutex);
    closeJob->threads.clear();
    closeJob->threadsReadyToShuffle.clear();
    closeJob->threadContextVec.clear();
    closeJob->shuffleRes.clear();
    delete closeJob->barrier;
    delete closeJob;
}