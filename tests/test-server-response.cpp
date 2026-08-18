#include "server-queue.h"

#include <cassert>
#include <memory>
#include <string>
#include <unordered_set>

struct test_result final : server_task_result {
    std::string value;

    json to_json() override {
        return value;
    }
};

static server_task_result_ptr make_result(int id, const std::string & value) {
    auto result = std::make_unique<test_result>();
    result->id = id;
    result->value = value;
    return result;
}

static void assert_result(
        const server_task_result_ptr & result,
        int expected_id,
        const std::string & expected_value) {
    assert(result);
    assert(result->id == expected_id);

    const auto * value = dynamic_cast<const test_result *>(result.get());
    assert(value);
    assert(value->value == expected_value);
}

int main() {
    server_response responses;
    responses.add_waiting_task_ids({ 1, 2 });

    {
        server_response_batch batch(responses);
        batch.send(make_result(1, "one-a"));
        batch.send(make_result(2, "two-a"));
        batch.send(make_result(99, "not-waiting"));
        batch.send(make_result(1, "one-b"));
        batch.send(make_result(2, "two-b"));
    }

    const std::unordered_set<int> ids = { 1, 2 };
    assert_result(responses.recv(ids), 1, "one-a");
    assert_result(responses.recv(ids), 2, "two-a");
    assert_result(responses.recv(ids), 1, "one-b");
    assert_result(responses.recv(ids), 2, "two-b");
    assert(!responses.recv_with_timeout({ 99 }, 0));

    // The existing scalar path remains available for the default-off control arm.
    responses.send(make_result(1, "scalar"));
    assert_result(responses.recv(1), 1, "scalar");

    // Explicit flush is reusable and destruction does not duplicate a response.
    {
        server_response_batch batch(responses);
        batch.send(make_result(2, "flush-a"));
        batch.flush();
        batch.send(make_result(2, "flush-b"));
    }
    assert_result(responses.recv(2), 2, "flush-a");
    assert_result(responses.recv(2), 2, "flush-b");

    return 0;
}
