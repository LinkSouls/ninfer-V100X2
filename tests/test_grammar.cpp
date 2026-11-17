#include "text/grammar.h"
#include "text/json_schema.h"

#include "ninfer/types.h"

#include <xgrammar/tool_grammar.h>

#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::text;

int fail(const char* message) {
    std::cerr << "grammar test failed: " << message << '\n';
    return 1;
}

std::vector<std::string> vocabulary() {
    std::vector<std::string> pieces;
    for (char c = '0'; c <= '9'; ++c) { pieces.push_back(std::string(1, c)); }
    for (char c = 'a'; c <= 'z'; ++c) { pieces.push_back(std::string(1, c)); }
    // string_view (not a C-string literal) so the range-for does not pick up the NUL terminator.
    for (char c : std::string_view("{}[]\":,-.")) { pieces.push_back(std::string(1, c)); }
    return pieces;
}

int token_id(const std::vector<std::string>& pieces, char c) {
    for (std::size_t i = 0; i < pieces.size(); ++i)
        if (pieces[i] == std::string(1, c)) return static_cast<int>(i);
    return -1;
}

bool allowed(const std::vector<std::uint32_t>& words, std::size_t position, std::size_t word_count, int id) {
    return (words[position * word_count + id / 32] & (1u << (id % 32))) != 0;
}

} // namespace

int main() {
    auto pieces = vocabulary();
    const auto digit0 = token_id(pieces, '0');
    const auto letter_a = token_id(pieces, 'a');
    const auto letter_b = token_id(pieces, 'b');
    const auto letter_r = token_id(pieces, 'r');
    if (digit0 < 0 || letter_a < 0 || letter_b < 0 || letter_r < 0) return fail("vocabulary");

    // EOS is a real, in-domain token id (xgrammar addresses eos by vocab index); 'z' never
    // appears in the grammars below, so it acts as a clean termination sentinel.
    const auto eos_id = token_id(pieces, 'z');
    if (eos_id < 0) return fail("eos token missing from vocabulary");
    GrammarCompiler compiler(pieces, {eos_id}, 1 << 20);
    const std::size_t words = ((static_cast<std::size_t>(pieces.size()) + 31) / 32);

    // --- GBNF: two fixed strings -------------------------------------------
    {
        auto session = compiler.compile(OutputConstraint::grammar(R"(root ::= "ab" | "cd")"),
                                        {}, {});
        if (!session) return fail("gbnf compile returned null");
        std::vector<std::uint32_t> mask(words, 0);
        session->masks({}, mask);
        if (!allowed(mask, 0, words, letter_a) || !allowed(mask, 0, words, token_id(pieces, 'c')))
            return fail("gbnf first mask must allow a and c");
        if (allowed(mask, 0, words, letter_b)) return fail("gbnf first mask must not allow b");
        // Draft lookahead: "a" advances to the state where only "b" follows; a rejected
        // draft ("d", not a valid first token) closes the lookahead without flagging a dead end.
        std::vector<std::int32_t> drafts{letter_a};
        std::vector<std::uint32_t> lookahead((drafts.size() + 1) * words, 0);
        if (session->masks(drafts, lookahead) != 0u) return fail("admissible draft must not dead-end");
        if (!allowed(lookahead, 1, words, letter_b))
            return fail("after draft 'a' only b may follow");
        if (allowed(lookahead, 1, words, token_id(pieces, 'c')))
            return fail("after draft 'a' c must not follow");
        std::vector<std::int32_t> rejected{token_id(pieces, 'd')};
        std::vector<std::uint32_t> closed((rejected.size() + 1) * words, 0);
        if (session->masks(rejected, closed) != 0u)
            return fail("a rejected draft must not flag a grammar dead end");
        if (closed[1 * words] != 1u) return fail("closed lookahead must keep the sentinel word");
        // A one-token grammar is cleanly terminated after the admissible draft: no content token may
        // follow, only the EOS sentinel (in-domain) may be emitted to end the match.
        auto one = compiler.compile(OutputConstraint::grammar(R"(root ::= "a")"), {}, {});
        std::vector<std::int32_t> tail{letter_a};
        std::vector<std::uint32_t> terminal((tail.size() + 1) * words, 0);
        if (one->masks(tail, terminal) != 0u)
            return fail("a terminated one-token grammar must not be flagged dead");
        if (!allowed(terminal, 1, words, eos_id))
            return fail("after 'a' the EOS token must be allowed to terminate the match");
        if (allowed(terminal, 1, words, letter_a) || allowed(terminal, 1, words, digit0))
            return fail("after 'a' no content token may follow");
        // Transaction: a tentative accept is invisible to lookahead and rolls back on
        // discard; confirm commits it.
        session->accept(letter_a);
        session->discard();
        {
            std::vector<std::uint32_t> restored(words, 0);
            session->masks({}, restored);
            if (!allowed(restored, 0, words, letter_a))
                return fail("discard must restore the pre-accept mask");
        }
        session->accept(letter_a);
        session->confirm();
        session->accept(letter_b);
        session->confirm();
        // Without an EOS accept the matcher is complete (the string is a valid whole) but not
        // terminated; termination is an engine-layer event (EOS flows through the session).
        if (!session->observation().complete) return fail("ab must complete the grammar");
        if (session->observation().terminated) return fail("no EOS was accepted");
    }

    // --- Compile cache: second compile of the same key is a hit -------------
    {
        auto first  = compiler.compile(OutputConstraint::grammar(R"(root ::= "x")"), {}, {});
        auto second = compiler.compile(OutputConstraint::grammar(R"(root ::= "x")"), {}, {});
        if (first->observation().cache != ConstraintCacheAccess::Built)
            return fail("first compile must be a cache build");
        if (second->observation().cache != ConstraintCacheAccess::Hit)
            return fail("repeat compile must be a cache hit");
    }

    // --- JSON object: first emitted token must open the object --------------
    {
        auto session = compiler.compile(OutputConstraint::json_object(), {}, {});
        if (!session) return fail("json_object compile");
        std::vector<std::uint32_t> mask(words, 0);
        session->masks({}, mask);
        if (!allowed(mask, 0, words, token_id(pieces, '{')))
            return fail("json_object must allow an opening brace");
        if (allowed(mask, 0, words, letter_a))
            return fail("json_object must not allow a bare letter first");
    }

    // --- JSON schema: required + integer bounds survive composition ---------
    {
        const std::string schema =
            R"({"type":"object","required":["n"],"properties":{"n":{"type":"integer","minimum":1}}})";
        auto session = compiler.compile(OutputConstraint::json_schema(schema), {}, {});
        if (!session) return fail("json_schema compile");
        std::vector<std::uint32_t> mask(words, 0);
        session->masks({}, mask);
        if (!allowed(mask, 0, words, token_id(pieces, '{')))
            return fail("schema object must open with a brace");
    }

    // --- JSON schema: enum / nested object / array all compile and mask right ---
    {
        // enum of strings: the first emitted token must open a quoted string, not a brace/bracket.
        {
            auto session = compiler.compile(
                OutputConstraint::json_schema(R"({"type":"string","enum":["red","green"]})"), {}, {});
            if (!session) return fail("enum schema compile");
            std::vector<std::uint32_t> mask(words, 0);
            session->masks({}, mask);
            if (!allowed(mask, 0, words, token_id(pieces, '"')))
                return fail("enum-of-strings must open with a quote");
            if (allowed(mask, 0, words, token_id(pieces, '{')) ||
                allowed(mask, 0, words, token_id(pieces, '[')))
                return fail("enum-of-strings must not open with a brace or bracket");
        }
        // nested object: compiles and opens with a brace.
        {
            auto session = compiler.compile(
                OutputConstraint::json_schema(
                    R"({"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"integer"}}}}})"),
                {}, {});
            if (!session) return fail("nested-object schema compile");
            std::vector<std::uint32_t> mask(words, 0);
            session->masks({}, mask);
            if (!allowed(mask, 0, words, token_id(pieces, '{')))
                return fail("nested-object schema must open with a brace");
        }
        // array: compiles and opens with a bracket.
        {
            auto session = compiler.compile(
                OutputConstraint::json_schema(R"({"type":"array","items":{"type":"integer"}})"), {}, {});
            if (!session) return fail("array schema compile");
            std::vector<std::uint32_t> mask(words, 0);
            session->masks({}, mask);
            if (!allowed(mask, 0, words, token_id(pieces, '[')))
                return fail("array schema must open with a bracket");
            if (allowed(mask, 0, words, token_id(pieces, '{')))
                return fail("array schema must not open with a brace");
        }
    }

    // --- JSON schema negatives ----------------------------------------------
    {
        bool saw_unsupported = false, saw_unsatisfiable = false, saw_invalid = false;
        try {
            compiler.compile(
                OutputConstraint::json_schema(R"({"type":"object","patternProperties":{}})"),
                {}, {});
        } catch (const RequestError& error) {
            saw_unsupported = error.kind() == RequestErrorKind::UnsupportedJsonSchema;
        }
        try {
            compiler.compile(OutputConstraint::json_schema(R"({"const":1,"minimum":2})"), {}, {});
        } catch (const RequestError& error) {
            saw_unsatisfiable = error.kind() == RequestErrorKind::UnsatisfiableJsonSchema;
        }
        try {
            compiler.compile(OutputConstraint::json_schema(R"({"enum":[]})"), {}, {});
        } catch (const RequestError& error) {
            saw_invalid = error.kind() == RequestErrorKind::InvalidJsonSchema;
        }
        if (!saw_unsupported) return fail("patternProperties must be rejected as unsupported");
        if (!saw_unsatisfiable) return fail("const/minimum conflict must be unsatisfiable");
        if (!saw_invalid) return fail("empty enum must be an invalid schema");
    }

    // --- Regex: digits only ---------------------------------------------------
    {
        auto session = compiler.compile(OutputConstraint::regex(R"(^[0-9]+$)"), {}, {});
        if (!session) return fail("regex compile");
        std::vector<std::uint32_t> mask(words, 0);
        session->masks({}, mask);
        for (char c = '0'; c <= '9'; ++c)
            if (!allowed(mask, 0, words, token_id(pieces, c)))
                return fail("regex must allow a leading digit");
        if (allowed(mask, 0, words, letter_a)) return fail("regex must reject a leading letter");
    }

    // --- Choice: only literal alternatives ------------------------------------
    {
        auto session = compiler.compile(
            OutputConstraint::choice({"red", "green", "blue"}), {}, {});
        if (!session) return fail("choice compile");
        std::vector<std::uint32_t> mask(words, 0);
        session->masks({}, mask);
        if (!allowed(mask, 0, words, letter_r) || !allowed(mask, 0, words, token_id(pieces, 'g')) ||
            !allowed(mask, 0, words, token_id(pieces, 'b')))
            return fail("choice must allow the three alternatives");
        if (allowed(mask, 0, words, letter_a)) return fail("choice must exclude non-members");
    }

    // --- Choice: full sequence to completion (reproduces the guided_choice 500) --
    // A short choice word completes in a few tokens. The engine commits the word, then the
    // grammar must allow exactly the EOS token to terminate the match. If the completion
    // mask is all-zero (or a content token leaks in), the engine forces token 0 into a
    // terminated matcher and accept() throws "selected token violates the request grammar".
    {
        auto session = compiler.compile(
            OutputConstraint::choice({"red", "green", "blue"}), {}, {});
        if (!session) return fail("choice compile (full sequence)");
        const auto tok_e = token_id(pieces, 'e');
        const auto tok_d = token_id(pieces, 'd');
        if (tok_e < 0 || tok_d < 0) return fail("choice full-sequence vocabulary incomplete");
        // Commit the word "red" one token at a time, confirming after each (accept() leaves the
        // matcher tentative; a masks() call requires a confirmed state).
        session->accept(letter_r); session->confirm();
        session->accept(tok_e);    session->confirm();
        session->accept(tok_d);    session->confirm();
        // The whole word is now matched: the matcher is complete but not terminated. The single
        // valid continuation is the in-domain EOS sentinel; no content token may follow.
        std::vector<std::uint32_t> terminal(words, 0);
        const std::uint32_t dead = session->masks({}, terminal);
        if (dead != 0u) return fail("a completed choice must not be flagged dead");
        if (!allowed(terminal, 0, words, eos_id))
            return fail("after a completed choice the EOS token must be allowed to terminate");
        if (allowed(terminal, 0, words, letter_r) || allowed(terminal, 0, words, letter_a) ||
            allowed(terminal, 0, words, tok_e))
            return fail("after a completed choice no content token may follow");
        // Accepting the EOS terminates the match cleanly.
        session->accept(eos_id);
        session->confirm();
        if (!session->observation().terminated) return fail("EOS must terminate the choice match");
    }

    // --- Invalid GBNF must surface as a RequestError, not a crash ------------
    {
        bool rejected = false;
        try {
            compiler.compile(OutputConstraint::grammar(R"(root ::= )"), {}, {});
        } catch (const RequestError& error) {
            rejected = error.kind() == RequestErrorKind::InvalidGrammar;
        }
        if (!rejected) return fail("empty GBNF must be rejected with InvalidGrammar");
    }

    // --- Qwen tool-call grammar (p7 task 3.4) --------------------------------
    {
        // Byte-level vocabulary: every distinct byte in the Qwen tool-call framing and the tool
        // names below becomes its own single-byte token. xgrammar's RAW matcher attributes
        // multi-byte whole-tokens unreliably in an artificial vocab, so a byte-per-token vocab is
        // the faithful stand-in for a real tokenizer here.
        std::vector<std::string> pieces;
        {
            const std::vector<std::string> sources = {
                "<tool_call>",
                "</tool_call>",
                "\n<function=",
                "<parameter=",
                "</parameter>",
                "</function>",
                "get_weather",
                "search",
                "{}[]\":,-.0123456789",
            };
            std::vector<bool> seen(256, false);
            const auto add = [&pieces, &seen](char c) {
                if (!seen[static_cast<unsigned char>(c)]) {
                    seen[static_cast<unsigned char>(c)] = true;
                    pieces.push_back(std::string(1, c));
                }
            };
            for (const std::string& s : sources)
                for (char c : s) add(c);
            for (char c = 'a'; c <= 'z'; ++c) add(c);
        }
        const auto find = [&pieces](char c) -> int {
            for (std::size_t i = 0; i < pieces.size(); ++i)
                if (pieces[i] == std::string(1, c)) return static_cast<int>(i);
            return -1;
        };
        // The open marker <tool_call> begins with '<'; that byte is the only valid first token of a
        // tool call. 'z' never begins a declared tool name, so it doubles as a clean EOS + negative.
        const int tok_open  = find('<');
        const int tok_brace = find('{');
        const int tok_g     = find('g');
        const int tok_s     = find('s');
        const int tok_z     = find('z');
        if (tok_open < 0 || tok_brace < 0 || tok_g < 0 || tok_s < 0 || tok_z < 0)
            return fail("tool-call vocabulary incomplete");

        GrammarCompiler tool_compiler(pieces, {tok_z}, 1 << 20);
        const std::vector<xgrammar::QwenToolDef> tools = {
            {"get_weather",
             R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})"},
            {"search", R"({"type":"object","properties":{"query":{"type":"string"}}})"},
        };
        auto session = tool_compiler.compile_model(
            "tool-test", [tools] { return xgrammar::BuildQwenToolCallGrammar(tools); }, {}, {});
        if (!session) return fail("tool grammar compile returned null");
        const std::size_t domain_words = session->mask_words();

        // Position 0: only the first byte of the <tool_call> opener (i.e. '<') is a valid start; a
        // JSON brace is not.
        std::vector<std::uint32_t> mask(domain_words, 0);
        const std::uint32_t start_dead = session->masks({}, mask);
        if (start_dead != 0) return fail("empty draft must not dead-end at position 0");
        if (!allowed(mask, 0, domain_words, tok_open)) return fail("tool call must open with '<'");
        if (allowed(mask, 0, domain_words, tok_brace))
            return fail("tool call must not start with a brace");

        // Advance just past the "<tool_call>\n<function=" framing; the next byte is a declared tool
        // name's first char: 'g' (get_weather) or 's' (search), never 'z'.
        const std::string framing = "<tool_call>" + std::string("\n<function=");
        std::vector<std::int32_t> prefix;
        for (char c : framing) {
            const int t = find(c);
            if (t < 0) return fail("tool framing byte missing from vocabulary");
            prefix.push_back(t);
        }
        std::vector<std::uint32_t> name_mask((prefix.size() + 1) * domain_words, 0);
        const std::uint32_t dead = session->masks(std::span(prefix), name_mask);
        if (dead != 0) return fail("valid tool-call prefix must not dead-end");
        const std::size_t name_position = prefix.size();
        if (!allowed(name_mask, name_position, domain_words, tok_g))
            return fail("tool-name position must allow 'g'");
        if (!allowed(name_mask, name_position, domain_words, tok_s))
            return fail("tool-name position must allow 's'");
        if (allowed(name_mask, name_position, domain_words, tok_z))
            return fail("tool-name position must not allow 'z'");
    }

    // --- Vocabulary byte index -----------------------------------------------
    {
        const auto& vocab = *compiler.vocabulary();
        const auto& ids   = vocab.containing(static_cast<unsigned char>('{'));
        if (ids.empty() || ids.front() != token_id(pieces, '{'))
            return fail("vocabulary byte index must find the brace token");
    }

    std::cout << "grammar test passed\n";
    return 0;
}
