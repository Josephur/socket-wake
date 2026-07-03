/* SPDX-License-Identifier: Apache-2.0 */
#ifndef SOCKET_WAKE_H
#define SOCKET_WAKE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The opaque detector handle. Its definition lives in the runtime; treat
 * as a pointer. Lifetime: call socket_wake_create once, feed/detect any
 * number of times, then socket_wake_destroy. The weights buffer passed
 * to create() must outlive the detector (see the FFI contract). */
typedef struct socket_wake_detector_internal socket_wake_detector_t;

/* Create a detector from a weights blob.
 *   weights:        pointer to the bytes of the .bin file (see DESIGN.md)
 *   weights_bytes:  length of the blob in bytes
 *   sample_rate_hz: must be 16000 (the runtime's only supported rate in v1)
 * Returns NULL on invalid magic/version or unsupported sample rate.
 * The weights buffer is borrowed, not copied. */
socket_wake_detector_t *socket_wake_create(const void *weights,
                                            size_t weights_bytes,
                                            size_t sample_rate_hz);

/* Append PCM samples to the running detector. Buffers internally until a
 * mel frame is available, then runs the CNN and updates the state machine.
 * Safe to call with samples=0 (no-op). */
void socket_wake_feed(socket_wake_detector_t *d,
                       const short *pcm, size_t samples);

/* Returns true exactly once per wake event. After handling, the caller
 * should call socket_wake_reset() to re-arm. */
bool socket_wake_detected(socket_wake_detector_t *d);

/* Re-arm the state machine so it can fire again. */
void socket_wake_reset(socket_wake_detector_t *d);

/* Free the detector. The borrowed weights buffer is NOT freed. */
void socket_wake_destroy(socket_wake_detector_t *d);

/* Memory introspection (for the memory-profile CI test). Returns the
 * peak RAM observed so far in bytes. */
size_t socket_wake_peak_ram_bytes(const socket_wake_detector_t *d);

/* Returns the size of the weights blob currently held by the detector. */
size_t socket_wake_weights_bytes(const socket_wake_detector_t *d);

#ifdef __cplusplus
}
#endif

#endif /* SOCKET_WAKE_H */