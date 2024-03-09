// !!! This is a file automatically generated by hipify!!!
#pragma once

#include <cstdint>
#include <utility>

#include <hip/hip_runtime_api.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/hip/HIPFunctions.h>
#include <c10/util/Exception.h>

/*
 * Stream pool note.
 *
 * A HIPStream is an abstraction of an actual cuStream on the GPU. HIPStreams
 * are backed by cuStreams, but they use several pools to minimize the costs
 * associated with creating, retaining, and destroying cuStreams.
 *
 * There are three pools per device, and a device's pools are lazily created.
 *
 * The first pool contains only the default stream. When the default stream
 * is requested it's returned.
 *
 * The second pool is the "low priority" or "default priority" streams. In
 * HIP builds there is no distinction between streams in this pool and streams
 * in the third pool (below). There are 32 of these streams per device, and
 * when a stream is requested one of these streams is returned round-robin.
 * That is, the first stream requested is at index 0, the second at index 1...
 * to index 31, then index 0 again.
 *
 * This means that if 33 low priority streams are requested, the first and
 * last streams requested are actually the same stream (under the covers)
 * and kernels enqueued on them cannot run concurrently.
 *
 * The third pool is the "high priority" streams. The third pool acts like
 * the second pool except the streams are created with a higher priority.
 *
 * These pools suggest that stream users should prefer many short-lived streams,
 * as the cost of acquiring and releasing streams is effectively zero. If
 * many longer-lived streams are required in performance critical scenarios
 * then the functionality here may need to be extended to allow, for example,
 * "reserving" a subset of the pool so that other streams do not accidentally
 * overlap the performance critical streams.
 *
 * Note: although the notion of "current stream for device" is thread local
 * (every OS thread has a separate current stream, as one might expect),
 * the stream pool is global across all threads; stream 0 is always stream 0
 * no matter which thread you use it on.  Multiple threads can synchronize
 * on the same stream.  Although the HIP documentation is not very clear
 * on the matter, streams are thread safe; e.g., it is safe to enqueue
 * a kernel on the same stream from two different threads.
 */

namespace c10::hip {

static constexpr int max_compile_time_stream_priorities = 4;

// Value object representing a HIP stream.  This is just a wrapper
// around c10::Stream, but it comes with a little extra HIP-specific
// functionality (conversion to hipStream_t), and a guarantee that
// the wrapped c10::Stream really is a HIP stream.
class C10_HIP_API HIPStream {
 public:
  enum Unchecked { UNCHECKED };

  /// Construct a HIPStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a HIP stream.
  explicit HIPStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::HIP);
  }

  /// Construct a HIPStream from a Stream with no error checking.
  /// This constructor uses the "named" constructor idiom, and can
  /// be invoked as: HIPStream(HIPStream::UNCHECKED, stream)
  explicit HIPStream(Unchecked, Stream stream) : stream_(stream) {}

  bool operator==(const HIPStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const HIPStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Implicit conversion to hipStream_t.
  operator hipStream_t() const {
    return stream();
  }

  /// Implicit conversion to Stream (a.k.a., forget that the stream is a
  /// HIP stream).
  operator Stream() const {
    return unwrap();
  }

  /// Used to avoid baking in device type explicitly to Python-side API.
  DeviceType device_type() const {
    return DeviceType::HIP;
  }

  /// Get the HIP device index that this stream is associated with.
  DeviceIndex device_index() const {
    return stream_.device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a HIP device.
  Device device() const {
    return Device(DeviceType::HIP, device_index());
  }

  /// Return the stream ID corresponding to this particular stream.
  StreamId id() const {
    return stream_.id();
  }

  bool query() const {
    DeviceGuard guard{stream_.device()};
    hipError_t err = C10_HIP_ERROR_HANDLED(hipStreamQuery(stream()));

    if (err == hipSuccess) {
      return true;
    } else if (err != hipErrorNotReady) {
      C10_HIP_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)hipGetLastError();
    }

    return false;
  }

  void synchronize() const {
    DeviceGuard guard{stream_.device()};
    c10::hip::stream_synchronize(stream());
  }

  int priority() const {
    DeviceGuard guard{stream_.device()};
    int priority = 0;
    C10_HIP_CHECK(hipStreamGetPriority(stream(), &priority));
    return priority;
  }

  /// Explicit conversion to hipStream_t.
  hipStream_t stream() const;

  /// Explicit conversion to Stream.
  Stream unwrap() const {
    return stream_;
  }

  /// Reversibly pack a HIPStream into a struct representation.
  /// Previously the stream's data was packed into a single int64_t,
  /// as it was assumed the fields would not require more than
  /// 64 bits of storage in total.
  /// See https://github.com/pytorch/pytorch/issues/75854
  /// for more information regarding newer platforms that may violate
  /// this assumption.
  ///
  /// The HIPStream can be unpacked using unpack().
  struct c10::StreamData3 pack3() const {
    return stream_.pack3();
  }

  // Unpack a HIPStream from the 3 fields generated by pack().
  static HIPStream unpack3(
      StreamId stream_id,
      DeviceIndex device_index,
      DeviceType device_type) {
    return HIPStream(Stream::unpack3(stream_id, device_index, device_type));
  }

  static std::tuple<int, int> priority_range() {
    // Note: this returns the range of priority **supported by PyTorch**, not
    // the range of priority **supported by HIP**. The former is a subset of
    // the latter.
    int least_priority = 0, greatest_priority = 0;
    C10_HIP_CHECK(
        hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
#ifdef USE_ROCM
    // See Note [HIP stream priorities]
    TORCH_INTERNAL_ASSERT(
        least_priority == 1, "Unexpected HIP stream priority range");
    least_priority = 0;
#else
    TORCH_INTERNAL_ASSERT(
        least_priority == 0, "Unexpected HIP stream priority range");
#endif
    TORCH_INTERNAL_ASSERT(
        greatest_priority <= -1, "Unexpected HIP stream priority range");
    greatest_priority = std::max(
        -c10::hip::max_compile_time_stream_priorities + 1, greatest_priority);
    return std::make_tuple(least_priority, greatest_priority);
  }

  // Deleted for now; use HIPEvent::block instead
  // void synchronize_with(const HIPEvent& event) const;

 private:
  Stream stream_;
};

/**
 * Get a new stream from the HIP stream pool.  You can think of this
 * as "creating" a new stream, but no such creation actually happens;
 * instead, streams are preallocated from the pool and returned in a
 * round-robin fashion.
 *
 * You can request a stream from the high priority pool by setting
 * isHighPriority to true, or a stream for a specific device by setting device
 * (defaulting to the current HIP stream.)
 */
C10_API HIPStream
getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);
// no default priority to disambiguate overloads
C10_API HIPStream
getStreamFromPool(const int priority, DeviceIndex device = -1);

/**
 * Get a HIPStream from a externally allocated one.
 *
 * This is mainly for interoperability with different libraries where we
 * want to operate on a non-torch allocated stream for data exchange or similar
 * purposes
 */
C10_API HIPStream
getStreamFromExternal(hipStream_t ext_stream, DeviceIndex device_index);

/**
 * Get the default HIP stream, for the passed HIP device, or for the
 * current device if no device index is passed.  The default stream is
 * where most computation occurs when you aren't explicitly using
 * streams.
 */
C10_API HIPStream getDefaultHIPStream(DeviceIndex device_index = -1);

/**
 * Get the current HIP stream, for the passed HIP device, or for the
 * current device if no device index is passed.  The current HIP stream
 * will usually be the default HIP stream for the device, but it may
 * be different if someone called 'setCurrentHIPStream' or used 'StreamGuard'
 * or 'HIPStreamGuard'.
 */
C10_API HIPStream getCurrentHIPStream(DeviceIndex device_index = -1);

/**
 * Set the current stream on the device of the passed in stream to be
 * the passed in stream.  Yes, you read that right: this function
 * has *nothing* to do with the current device: it toggles the current
 * stream of the device of the passed stream.
 *
 * Confused?  Avoid using this function; prefer using 'HIPStreamGuard' instead
 * (which will switch both your current device and current stream in the way you
 * expect, and reset it back to its original state afterwards).
 */
C10_API void setCurrentHIPStream(HIPStream stream);

C10_API std::ostream& operator<<(std::ostream& stream, const HIPStream& s);

} // namespace c10::hip

namespace std {
template <>
struct hash<c10::hip::HIPStream> {
  size_t operator()(c10::hip::HIPStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std