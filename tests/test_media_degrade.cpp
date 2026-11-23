#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ContentPart text_part(std::string text) {
    ContentPart part;
    part.kind = ContentKind::Text;
    part.text = std::move(text);
    return part;
}

ContentPart image_part() {
    ContentPart part;
    part.kind     = ContentKind::Image;
    part.type_raw = "image_url";
    return part;
}

ContentPart video_part() {
    ContentPart part;
    part.kind     = ContentKind::Video;
    part.type_raw = "video_url";
    return part;
}

ChatTurn user_turn(std::vector<ContentPart> parts) {
    ChatTurn turn;
    turn.role    = ChatRole::User;
    turn.content = std::move(parts);
    return turn;
}

ChatTurn assistant_turn(std::vector<ContentPart> parts) {
    ChatTurn turn;
    turn.role    = ChatRole::Assistant;
    turn.content = std::move(parts);
    return turn;
}

} // namespace

int main() {
    int failures = 0;

    // History media (2 images + 1 video) is rewritten to the exact placeholders;
    // turn structure is preserved.
    {
        GenerationRequest request;
        request.messages = {
            user_turn({text_part("look at this"), image_part(), image_part()}),
            assistant_turn({text_part("two pictures")}),
            user_turn({video_part(), text_part("and this video")}),
            user_turn({text_part("what else?")}),
        };
        const std::size_t degraded = degrade_media_history(request);
        failures += check(degraded == 3, "expected 3 degraded media parts (2 images + 1 video)");
        failures += check(request.messages.size() == 4, "turn count changed by degradation");
        failures += check(request.messages[0].content.size() == 3,
                          "media-only part count must be preserved in the turn");
        failures += check(request.messages[0].content[0].kind == ContentKind::Text,
                          "first part must stay text");
        failures += check(request.messages[0].content[1].text ==
                              std::string(kImageOmittedPlaceholder),
                          "image placeholder must match the fixed contract text");
        failures += check(request.messages[0].content[2].text ==
                              std::string(kImageOmittedPlaceholder),
                          "second image placeholder must match the fixed contract text");
        failures += check(request.messages[2].content[0].text ==
                              std::string(kVideoOmittedPlaceholder),
                          "video placeholder must match the fixed contract text");
        failures += check(request.messages[2].content[0].kind == ContentKind::Text,
                          "degraded video part must be a text part");
        failures += check(request.messages[2].content[1].text == "and this video",
                          "text parts in a degraded turn must be untouched");
        failures += check(request.media_item_count() == 0,
                          "after degradation no media parts may remain");
    }

    // Media on the current turn (last user message) is NOT degraded, so the
    // vision_disabled 400 still fires.
    {
        GenerationRequest request;
        request.messages = {
            user_turn({image_part(), text_part("old image")}),
            user_turn({text_part("now look"), image_part()}),
        };
        const std::size_t degraded = degrade_media_history(request);
        failures += check(degraded == 1, "only the history image may be degraded");
        failures += check(request.messages[1].content[1].kind == ContentKind::Image,
                          "media on the current turn must stay an image part");
        failures += check(request.media_item_count() == 1,
                          "the current-turn image must still count as media");
    }

    // A media-only history turn is not dropped: it survives as a placeholder turn.
    {
        GenerationRequest request;
        request.messages = {
            user_turn({image_part()}),
            user_turn({text_part("what was that?")}),
        };
        const std::size_t degraded = degrade_media_history(request);
        failures += check(degraded == 1, "media-only history turn must degrade its part");
        failures += check(request.messages.size() == 2, "media-only history turn must survive");
        failures += check(request.messages[0].content.size() == 1 &&
                              request.messages[0].content[0].text ==
                                  std::string(kImageOmittedPlaceholder),
                          "media-only history turn must become a placeholder text turn");
    }

    // A turn after the last user message is history, not the current turn.
    {
        GenerationRequest request;
        request.messages = {
            user_turn({text_part("describe")}),
            assistant_turn({image_part()}),
            user_turn({text_part("ok now the second one")}),
        };
        const std::size_t degraded = degrade_media_history(request);
        failures += check(degraded == 1, "assistant-turn media must be degraded as history");
        failures += check(request.messages[1].content[0].kind == ContentKind::Text,
                          "assistant media part must become the placeholder text");
    }

    // No user turn: every turn is history.
    {
        GenerationRequest request;
        request.messages = {assistant_turn({image_part()})};
        const std::size_t degraded = degrade_media_history(request);
        failures += check(degraded == 1, "a request without a user turn degrades all media");
    }

    // Pure-text request: nothing degrades, nothing mutates.
    {
        GenerationRequest request;
        request.messages = {user_turn({text_part("hello")})};
        const std::size_t degraded = degrade_media_history(request);
        failures += check(degraded == 0, "a text-only request must degrade nothing");
        failures += check(request.messages[0].content[0].text == "hello",
                          "a text-only request must be left untouched");
    }

    if (failures == 0) {
        std::cout << "media degrade tests passed\n";
    }
    return failures;
}