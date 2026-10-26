// Native timing submission avoids Python gaps between the two device events.
#include <acl/acl.h>

extern "C" int pp_launch_measured(void *function, unsigned int blocks, void *stream,
                                  void *args, void *start, void *end, float *milliseconds)
{
    auto functionHandle = reinterpret_cast<aclrtFuncHandle>(function);
    auto streamHandle = reinterpret_cast<aclrtStream>(stream);
    auto argsHandle = reinterpret_cast<aclrtArgsHandle>(args);
    auto startEvent = reinterpret_cast<aclrtEvent>(start);
    auto endEvent = reinterpret_cast<aclrtEvent>(end);
    aclError status = aclrtRecordEvent(startEvent, streamHandle);
    if (status != ACL_SUCCESS) { return status; }
    status = aclrtLaunchKernelWithConfig(functionHandle, blocks, streamHandle, nullptr, argsHandle, nullptr);
    if (status != ACL_SUCCESS) { return status; }
    status = aclrtRecordEvent(endEvent, streamHandle);
    if (status != ACL_SUCCESS) { return status; }
    status = aclrtSynchronizeStream(streamHandle);
    if (status != ACL_SUCCESS) { return status; }
    return aclrtEventElapsedTime(milliseconds, startEvent, endEvent);
}
