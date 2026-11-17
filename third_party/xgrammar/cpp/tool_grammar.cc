/*!
 * \file xgrammar/tool_grammar.cc
 * \brief Qwen XML tool-call grammar: framing + per-tool arguments + choice.
 */

#include "xgrammar/tool_grammar.h"

#include "grammar_functor.h"
#include "json_schema_converter.h"

#include <utility>

namespace xgrammar {

namespace {

// Escape a literal into a valid EBNF string literal (the lexer interprets \n, \t, \", \\).
std::string EbnfLiteral(const std::string& value) {
  std::string out = "\"";
  for (const char c : value) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  out += "\"";
  return out;
}

}  // namespace

Grammar BuildQwenToolCallGrammar(const std::vector<QwenToolDef>& tools) {
  if (tools.empty()) {
    throw JSONSchemaCompileError(SchemaErrorType::kInvalidSchema, "no tools declared");
  }
  std::vector<Grammar> framed;
  framed.reserve(tools.size());
  for (const auto& tool : tools) {
    if (tool.name.empty()) {
      throw JSONSchemaCompileError(SchemaErrorType::kInvalidSchema, "tool name is empty");
    }
    const std::string params =
        tool.parameters_json.empty() ? R"({"type":"object"})" : tool.parameters_json;
    // Outer object as <parameter> blocks (Qwen XML); inner values stay standard JSON.
    auto args = JSONSchemaToGrammar(
        params, /*any_whitespace=*/true, std::nullopt,
        std::pair<std::string, std::string>{",", ":"}, /*strict_mode=*/true, std::nullopt,
        /*any_order=*/false, JSONFormat::kQwenXML, {});
    auto prefix =
        Grammar::FromEBNF("root ::= " + EbnfLiteral("<tool_call>\n<function=" + tool.name + ">\n"));
    auto suffix = Grammar::FromEBNF("root ::= " + EbnfLiteral("</function>\n"));
    framed.push_back(Grammar::Concat({prefix, args, suffix}));
  }
  auto grammar = framed.size() == 1 ? framed.front() : Grammar::Union(framed);
  return GrammarNormalizer::Apply(std::move(grammar));
}

}  // namespace xgrammar