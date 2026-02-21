//! Deterministic output checks for audit evaluation.
//!
//! Fast, pure-Rust checks that verify structural properties of model
//! outputs without requiring a model forward pass. These are sanity checks
//! — not full parsers — designed to run in < 1ms per inference.

use crate::audit::types::DeterministicCheck;

// ─── Category Detection ─────────────────────────────────────────────────────

/// Detect the task category from an input string via keyword matching.
///
/// Returns one of: `"code_generation"`, `"json"`, `"sql"`, `"math"`, `"qa"`, `"general"`.
pub fn detect_category(input: &str) -> String {
    let lower = input.to_lowercase();

    if lower.contains("write code")
        || lower.contains("implement")
        || lower.contains("function")
        || lower.contains("def ")
        || lower.contains("fn ")
        || lower.contains("class ")
        || lower.contains("program")
    {
        "code_generation".to_string()
    } else if lower.contains("json") || lower.contains("schema") {
        "json".to_string()
    } else if lower.contains("sql")
        || lower.contains("select ")
        || lower.contains("query")
        || lower.contains("database")
    {
        "sql".to_string()
    } else if lower.contains("calculate")
        || lower.contains("compute")
        || lower.contains("solve")
        || lower.contains("math")
        || lower.contains("equation")
    {
        "math".to_string()
    } else if lower.contains("what is")
        || lower.contains("explain")
        || lower.contains("describe")
        || lower.contains("how does")
        || lower.contains("why ")
        || lower.contains("define")
    {
        "qa".to_string()
    } else {
        "general".to_string()
    }
}

// ─── Main Entry ─────────────────────────────────────────────────────────────

/// Run all applicable deterministic checks on an inference output.
///
/// The `task_hint` (if provided) or auto-detected category determines which
/// checks are run. Returns a list of `DeterministicCheck` results.
pub fn evaluate_deterministic(
    input: &str,
    output: &str,
    task_hint: Option<&str>,
) -> Vec<DeterministicCheck> {
    let category = task_hint
        .map(String::from)
        .unwrap_or_else(|| detect_category(input));
    let mut checks = Vec::new();

    // Always: non-empty output.
    checks.push(check_non_empty(output));

    // Category-specific checks.
    match category.as_str() {
        "json" => checks.push(check_json(output)),
        "code_generation" => checks.push(check_code_syntax(output)),
        "sql" => checks.push(check_sql(output)),
        "math" => checks.push(check_math(output)),
        _ => {}
    }

    // Always: structured output (truncation detection).
    checks.push(check_structured_output(output));

    checks
}

// ─── Individual Checks ──────────────────────────────────────────────────────

fn check_non_empty(output: &str) -> DeterministicCheck {
    let trimmed = output.trim();
    DeterministicCheck {
        check_type: "non_empty".to_string(),
        passed: !trimmed.is_empty(),
        detail: if trimmed.is_empty() {
            Some("Output is empty or whitespace-only".to_string())
        } else {
            None
        },
    }
}

fn check_json(output: &str) -> DeterministicCheck {
    let trimmed = output.trim();
    let json_str = extract_json_block(trimmed).unwrap_or(trimmed);

    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(_) => DeterministicCheck {
            check_type: "json_valid".to_string(),
            passed: true,
            detail: None,
        },
        Err(e) => DeterministicCheck {
            check_type: "json_valid".to_string(),
            passed: false,
            detail: Some(format!("JSON parse error: {}", e)),
        },
    }
}

fn check_code_syntax(output: &str) -> DeterministicCheck {
    let code = extract_code_block(output).unwrap_or(output);

    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut string_char = '"';
    let mut prev = '\0';

    for ch in code.chars() {
        if in_string {
            if ch == string_char && prev != '\\' {
                in_string = false;
            }
            prev = ch;
            continue;
        }

        match ch {
            '"' | '\'' => {
                in_string = true;
                string_char = ch;
            }
            '(' | '[' | '{' => stack.push(ch),
            ')' => {
                if stack.last() != Some(&'(') {
                    return DeterministicCheck {
                        check_type: "code_syntax".to_string(),
                        passed: false,
                        detail: Some("Unmatched ')'".to_string()),
                    };
                }
                stack.pop();
            }
            ']' => {
                if stack.last() != Some(&'[') {
                    return DeterministicCheck {
                        check_type: "code_syntax".to_string(),
                        passed: false,
                        detail: Some("Unmatched ']'".to_string()),
                    };
                }
                stack.pop();
            }
            '}' => {
                if stack.last() != Some(&'{') {
                    return DeterministicCheck {
                        check_type: "code_syntax".to_string(),
                        passed: false,
                        detail: Some("Unmatched '}'".to_string()),
                    };
                }
                stack.pop();
            }
            _ => {}
        }
        prev = ch;
    }

    DeterministicCheck {
        check_type: "code_syntax".to_string(),
        passed: stack.is_empty(),
        detail: if stack.is_empty() {
            None
        } else {
            Some(format!("Unclosed brackets: {:?}", stack))
        },
    }
}

fn check_sql(output: &str) -> DeterministicCheck {
    let sql = extract_code_block(output)
        .unwrap_or(output)
        .trim()
        .to_uppercase();

    let has_keyword = sql.contains("SELECT")
        || sql.contains("INSERT")
        || sql.contains("UPDATE")
        || sql.contains("DELETE")
        || sql.contains("CREATE")
        || sql.contains("ALTER")
        || sql.contains("DROP");

    if !has_keyword {
        return DeterministicCheck {
            check_type: "sql_parse".to_string(),
            passed: false,
            detail: Some("No SQL keywords found".to_string()),
        };
    }

    // SELECT without FROM (unless simple expression).
    if sql.contains("SELECT") && !sql.contains("FROM") {
        let is_simple = sql.lines().count() == 1 && !sql.contains('*');
        if !is_simple {
            return DeterministicCheck {
                check_type: "sql_parse".to_string(),
                passed: false,
                detail: Some("SELECT without FROM clause".to_string()),
            };
        }
    }

    // Balanced parentheses.
    let open = sql.chars().filter(|&c| c == '(').count();
    let close = sql.chars().filter(|&c| c == ')').count();
    if open != close {
        return DeterministicCheck {
            check_type: "sql_parse".to_string(),
            passed: false,
            detail: Some(format!(
                "Unbalanced parentheses: {} open, {} close",
                open, close
            )),
        };
    }

    DeterministicCheck {
        check_type: "sql_parse".to_string(),
        passed: true,
        detail: None,
    }
}

fn check_math(output: &str) -> DeterministicCheck {
    // Look for simple "LHS = RHS" patterns and verify the arithmetic.
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(eq_pos) = trimmed.rfind('=') {
            let lhs = trimmed[..eq_pos].trim();
            let rhs = trimmed[eq_pos + 1..].trim();

            if let (Some(expected), Some(actual)) = (eval_simple_expr(lhs), parse_number(rhs)) {
                if (expected - actual).abs() > 0.001 {
                    return DeterministicCheck {
                        check_type: "math_correct".to_string(),
                        passed: false,
                        detail: Some(format!(
                            "Math error: {} = {} (expected {})",
                            lhs, rhs, expected
                        )),
                    };
                }
            }
        }
    }

    DeterministicCheck {
        check_type: "math_correct".to_string(),
        passed: true,
        detail: None,
    }
}

fn check_structured_output(output: &str) -> DeterministicCheck {
    let trimmed = output.trim();

    // Truncation indicators.
    let truncated = (trimmed.ends_with("...") && trimmed.len() > 100)
        || trimmed.ends_with("</s>")
        || trimmed.ends_with("<|endoftext|>");

    DeterministicCheck {
        check_type: "structured_output".to_string(),
        passed: !truncated,
        detail: if truncated {
            Some("Output appears truncated".to_string())
        } else {
            None
        },
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Extract JSON content from markdown code fences or raw JSON.
fn extract_json_block(s: &str) -> Option<&str> {
    // Try ```json ... ``` first.
    if let Some(start) = s.find("```json") {
        let content_start = start + 7;
        if let Some(end) = s[content_start..].find("```") {
            return Some(s[content_start..content_start + end].trim());
        }
    }

    // Try ``` ... ```.
    if let Some(start) = s.find("```") {
        let content_start = start + 3;
        let line_end = s[content_start..].find('\n').unwrap_or(0);
        let actual_start = content_start + line_end;
        if let Some(end) = s[actual_start..].find("```") {
            return Some(s[actual_start..actual_start + end].trim());
        }
    }

    // Try raw { ... } or [ ... ].
    let first_brace = s.find('{');
    let last_brace = s.rfind('}');
    let first_bracket = s.find('[');
    let last_bracket = s.rfind(']');

    match (first_brace, last_brace, first_bracket, last_bracket) {
        (Some(fb), Some(lb), _, _) if fb < lb => Some(&s[fb..=lb]),
        (_, _, Some(fk), Some(lk)) if fk < lk => Some(&s[fk..=lk]),
        _ => None,
    }
}

/// Extract code from markdown code fences.
fn extract_code_block(s: &str) -> Option<&str> {
    if let Some(start) = s.find("```") {
        let content_start = start + 3;
        let line_end = s[content_start..].find('\n').unwrap_or(0);
        let actual_start = content_start + line_end;
        if let Some(end) = s[actual_start..].find("```") {
            return Some(s[actual_start..actual_start + end].trim());
        }
    }
    None
}

/// Evaluate a simple arithmetic expression (+ - * / only, left to right).
fn eval_simple_expr(expr: &str) -> Option<f64> {
    let expr = expr.trim();
    if expr.is_empty() {
        return None;
    }

    let mut result = 0.0f64;
    let mut current_num = String::new();
    let mut op = '+';

    for (i, ch) in expr.chars().enumerate() {
        let is_op = (ch == '+' || ch == '-' || ch == '*' || ch == '/')
            && i > 0
            && !current_num.trim().is_empty();

        if is_op {
            let n = parse_number(current_num.trim())?;
            result = apply_op(result, n, op)?;
            current_num.clear();
            op = ch;
        } else {
            current_num.push(ch);
        }
    }

    // Last number.
    let n = parse_number(current_num.trim())?;
    apply_op(result, n, op)
}

fn apply_op(lhs: f64, rhs: f64, op: char) -> Option<f64> {
    match op {
        '+' => Some(lhs + rhs),
        '-' => Some(lhs - rhs),
        '*' => Some(lhs * rhs),
        '/' if rhs != 0.0 => Some(lhs / rhs),
        _ => None,
    }
}

fn parse_number(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_category() {
        assert_eq!(
            detect_category("Write code for a function"),
            "code_generation"
        );
        assert_eq!(
            detect_category("implement a binary search"),
            "code_generation"
        );
        assert_eq!(detect_category("Return the result as JSON"), "json");
        assert_eq!(detect_category("Write a SQL query to find users"), "sql");
        assert_eq!(detect_category("Calculate 2+2"), "math");
        assert_eq!(detect_category("What is photosynthesis?"), "qa");
        assert_eq!(detect_category("Tell me a joke"), "general");
        assert_eq!(detect_category("Explain the theory of relativity"), "qa");
        assert_eq!(detect_category("solve this equation"), "math");
        assert_eq!(detect_category("select all users from table"), "sql");
    }

    #[test]
    fn test_json_valid() {
        let checks = evaluate_deterministic("return json", r#"{"key": "value"}"#, Some("json"));
        let json_check = checks
            .iter()
            .find(|c| c.check_type == "json_valid")
            .unwrap();
        assert!(json_check.passed);
    }

    #[test]
    fn test_json_invalid() {
        let checks = evaluate_deterministic("return json", "{invalid json", Some("json"));
        let json_check = checks
            .iter()
            .find(|c| c.check_type == "json_valid")
            .unwrap();
        assert!(!json_check.passed);
        assert!(json_check.detail.is_some());
    }

    #[test]
    fn test_json_in_code_fence() {
        let output = "Here is the JSON:\n```json\n{\"name\": \"test\", \"value\": 42}\n```\nDone.";
        let checks = evaluate_deterministic("", output, Some("json"));
        let json_check = checks
            .iter()
            .find(|c| c.check_type == "json_valid")
            .unwrap();
        assert!(json_check.passed);
    }

    #[test]
    fn test_code_syntax_balanced() {
        let code = "```python\ndef foo(x):\n    return [x, {1: 2}]\n```";
        let checks = evaluate_deterministic("", code, Some("code_generation"));
        let syntax = checks
            .iter()
            .find(|c| c.check_type == "code_syntax")
            .unwrap();
        assert!(syntax.passed);
    }

    #[test]
    fn test_code_syntax_unbalanced() {
        let code = "function foo() { return [1, 2";
        let checks = evaluate_deterministic("", code, Some("code_generation"));
        let syntax = checks
            .iter()
            .find(|c| c.check_type == "code_syntax")
            .unwrap();
        assert!(!syntax.passed);
    }

    #[test]
    fn test_math_correct() {
        let checks = evaluate_deterministic("", "2 + 2 = 4", Some("math"));
        let math = checks
            .iter()
            .find(|c| c.check_type == "math_correct")
            .unwrap();
        assert!(math.passed);
    }

    #[test]
    fn test_math_wrong() {
        let checks = evaluate_deterministic("", "2 + 2 = 5", Some("math"));
        let math = checks
            .iter()
            .find(|c| c.check_type == "math_correct")
            .unwrap();
        assert!(!math.passed);
    }

    #[test]
    fn test_sql_valid() {
        let sql = "```sql\nSELECT * FROM users WHERE id = 1\n```";
        let checks = evaluate_deterministic("", sql, Some("sql"));
        let sql_check = checks.iter().find(|c| c.check_type == "sql_parse").unwrap();
        assert!(sql_check.passed);
    }

    #[test]
    fn test_sql_unbalanced_parens() {
        let checks = evaluate_deterministic("", "SELECT * FROM users WHERE (id = 1", Some("sql"));
        let sql_check = checks.iter().find(|c| c.check_type == "sql_parse").unwrap();
        assert!(!sql_check.passed);
    }

    #[test]
    fn test_non_empty_passes() {
        let checks = evaluate_deterministic("hi", "hello world", None);
        let ne = checks.iter().find(|c| c.check_type == "non_empty").unwrap();
        assert!(ne.passed);
    }

    #[test]
    fn test_non_empty_fails() {
        let checks = evaluate_deterministic("hi", "   ", None);
        let ne = checks.iter().find(|c| c.check_type == "non_empty").unwrap();
        assert!(!ne.passed);
    }

    #[test]
    fn test_structured_output_truncated() {
        let long = "a".repeat(200) + "...";
        let checks = evaluate_deterministic("", &long, None);
        let so = checks
            .iter()
            .find(|c| c.check_type == "structured_output")
            .unwrap();
        assert!(!so.passed);
    }

    #[test]
    fn test_all_checks_run_fast() {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            evaluate_deterministic(
                "Write a function to sort",
                "```python\ndef sort(arr):\n    return sorted(arr)\n```",
                None,
            );
        }
        let elapsed = start.elapsed();
        // 1000 evaluations should take < 100ms (< 0.1ms each).
        assert!(elapsed.as_millis() < 100, "Too slow: {:?}", elapsed);
    }
}
