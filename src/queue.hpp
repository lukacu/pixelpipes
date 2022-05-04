#pragma once

#include <thread>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <mutex>
#include <string>
#include <condition_variable>

#include <pixelpipes/base.hpp>

namespace pixelpipes {

class DispatchQueue {
	typedef Function<void(void)> Task;

public:
	DispatchQueue(size_t thread_cnt = 1);
	~DispatchQueue();

	// dispatch and copy
	void dispatch(const Task& op);
	// dispatch and move
	void dispatch(Task&& op);

	// Deleted operations
	DispatchQueue(const DispatchQueue& rhs) = delete;
	DispatchQueue& operator=(const DispatchQueue& rhs) = delete;
	DispatchQueue(DispatchQueue&& rhs) = delete;
	DispatchQueue& operator=(DispatchQueue&& rhs) = delete;

private:

	std::mutex lock_;
	std::vector<std::thread> threads_;
	std::queue<Task> q_;
	std::condition_variable cv_;
	bool quit_ = false;

	void dispatch_thread_handler(void);
};

}
