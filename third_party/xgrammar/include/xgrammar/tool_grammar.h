/*!
 * \file xgrammar/tool_grammar.h
 * \brief Model-owned tool-calling grammar for the Qwen XML tool-call framing.
 */

#ifndef XGRAMMAR_TOOL_GRAMMAR_H_
#define XGRAMMAR_TOOL_GRAMMAR_H_

#include <string>
#include <vector>

#include "xgrammar/grammar.h"

namespace xgrammar {

// One declared tool: its name and the JSON schema of its arguments.
struct QwenToolDef {
  std::string name;
  std::string parameters_json;
};

/*!
 * \brief Build a complete Qwen tool-call grammar.
 *
 * The result is a choice over the declared tools, each framed exactly as the Qwen
 * chat template renders an assistant tool call:
 *
 *     <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n
 *
 * The arguments object is rendered in the Qwen XML format (outer object as
 * `<parameter>` blocks, inner values as standard JSON), matching the template's
 * `render_tool_call`. A tool with no parameter schema is treated as an empty
 * object. The returned Grammar is normalized and its root is the tool-call choice.
 *
 * \param tools The declared tools; must be non-empty.
 * \throws JSONSchemaCompileError if a tool name is empty or a parameter schema is
 *         invalid/unsupported.
 */
Grammar BuildQwenToolCallGrammar(const std::vector<QwenToolDef>& tools);

}  // namespace xgrammar

#endif  // XGRAMMAR_TOOL_GRAMMAR_H_