# Documentation Writing Standards

## Core Principles

1. **Lead with value**: Start pages with what users can accomplish, not definitions
2. **Tables first**: Use tables for all structured data (properties, options, comparisons)
3. **Single source of truth**: Link to canonical docs instead of duplicating content
4. **Progressive disclosure**: Essential info first, details in accordions
5. **Concise prose**: Remove filler words, merge redundant paragraphs

## Page Structure

### Title and Description

Start each page with a clear title and 1-sentence description of what the page enables.

```markdown
---
title: For Loop Node
description: Iterating over multiple values in automations
---

Use `for_loop_node` to repeat actions for each value in a list—processing search results, downloading multiple files, or clicking through items.
```

### Section Organization

Use descriptive headers without numbers. Organize hierarchically:

```markdown
## Overview

## Properties

### Property Details

## Examples
```

**Never use numbered headers** like "1. Getting Started" or "1.1 Installation".

## Tables

### Use Tables For

- All property/parameter lists
- Feature comparisons
- Option summaries
- Quick references

### Property Table Format

```markdown
| Property  | Type          | Default | Description        |
| --------- | ------------- | ------- | ------------------ |
| `command` | `str \| None` | `None`  | Playwright locator |
| `xpath`   | `str \| None` | `None`  | XPath selector     |
```

### Comparison Table Format

```markdown
| Use Case            | Recommended             |
| ------------------- | ----------------------- |
| Tables, forms, text | `llm` with `axtree`     |
| Charts, images      | `llm` with `screenshot` |
```

## Code Examples

### Lead with Minimal Examples

Show the simplest working example first:

```json
{
    "interaction_action": {
        "click_element": {
            "command": "get_by_role(\"button\", name=\"Submit\")"
        }
    }
}
```

### Expand in Sections

Add complexity in dedicated sections or accordions.

### Code Block Language Tags

Always specify language: `json`, `python`, `bash`

## Callouts

### Limits

| Callout     | Max per page | Use for                     |
| ----------- | ------------ | --------------------------- |
| `<Info>`    | 1-2          | Version notes, plan info    |
| `<Tip>`     | 2-3          | Best practices, shortcuts   |
| `<Warning>` | Sparingly    | Breaking changes, data loss |

### Never Stack Callouts

Bad:

```markdown
<Info>Note 1</Info>
<Tip>Note 2</Tip>
```

Good: Combine into prose or use one callout.

## Accordions

Use `<AccordionGroup>` for:

- FAQs (each question as accordion title)
- Advanced configuration
- Troubleshooting
- Framework-specific variations

**Never use** for:

- Essential information users need upfront
- Quick start guides
- Basic setup

## Avoiding Redundancy

### Link, Don't Duplicate

When content exists elsewhere, link to it:

```markdown
See [Parameters](/docs/building-automations/parameters) for detailed usage.
```

### Single Canonical Answer

If multiple FAQs have the same answer, consolidate into one comprehensive answer with a table.

## Writing Style

### Voice and Tone

- Second person ("you")
- Active voice
- Clear, concise sentences
- Define jargon when necessary

### Be Specific

**Good:**

```markdown
Use `get_by_role("button", name="Submit")` for button clicks.
```

**Bad:**

```markdown
You can use various locator methods to find elements.
```

## FAQs Format

Group related questions with `<AccordionGroup>`:

```markdown
## Security

<AccordionGroup>
  <Accordion title="How do I handle passwords securely?">
    Use `secure_parameters` with 1Password or TOTP integration.

    | Provider | Use Case |
    |----------|----------|
    | 1Password | Passwords, API keys |
    | TOTP | 2FA codes |

  </Accordion>
</AccordionGroup>
```

## Docs Manifest

Every time you add a new documentation page, you **must** update `docs-manifest.json` in the repo root of the `docs/` folder. Add a new entry to the `documents` array with the following fields populated: `id`, `path`, `title`, `summary`, `gist`, `keywords`, `document_type`, `tags`, `relationships`, `entities`, `reading_priority`, `when_to_use`, and `embedding_hint`. Use the existing entries as a model for each field.

The `id` must be a kebab-case slug derived from the file path (e.g. `docs/action-types/foo.mdx` → `docs-action-types-foo`). The `path` is relative to the `docs/` folder root.

Failing to update the manifest means the new page will not be discoverable by the LLM navigation layer.

## Checklist Before Publishing

- [ ] Page starts with what users can accomplish
- [ ] Properties documented in tables
- [ ] Minimal example shown first
- [ ] No duplicate content (links to canonical sources)
- [ ] Callouts used sparingly and purposefully
- [ ] Code examples have language tags
- [ ] All terms are defined or linked
