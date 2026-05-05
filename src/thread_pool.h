#ifndef OEFP_THREAD_POOL_INTERNAL_H
#define OEFP_THREAD_POOL_INTERNAL_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace OEFP {
namespace detail {

inline void ParallelFor(
    std::size_t begin,
    std::size_t end,
    std::size_t chunk_size,
    std::size_t num_threads,
    const std::function<void(std::size_t, std::size_t)>& body) {
    if (end <= begin) {
        return;
    }

    const auto actual_chunk_size = std::max<std::size_t>(chunk_size, 1);
    const auto chunk_count = (end - begin + actual_chunk_size - 1) / actual_chunk_size;
    auto worker_count = num_threads;
    if (worker_count == 0) {
        worker_count = std::thread::hardware_concurrency();
    }
    worker_count = std::max<std::size_t>(worker_count, 1);
    worker_count = std::min(worker_count, chunk_count);

    std::atomic<std::size_t> next{begin};
    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&]() {
            while (true) {
                const auto chunk_begin = next.fetch_add(actual_chunk_size);
                if (chunk_begin >= end) {
                    return;
                }
                const auto chunk_end = std::min(chunk_begin + actual_chunk_size, end);

                try {
                    body(chunk_begin, chunk_end);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!first_exception) {
                        first_exception = std::current_exception();
                    }
                    return;
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

} // namespace detail
} // namespace OEFP

#endif // OEFP_THREAD_POOL_INTERNAL_H
