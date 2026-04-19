import re
import os

def find_brace_content(text, start_pos):
    """Finds the content of the brace starting at start_pos, handling nesting and escaped braces."""
    if start_pos >= len(text) or text[start_pos] != '{':
        return None, start_pos
    
    count = 0
    i = start_pos
    while i < len(text):
        char = text[i]
        if char == '\\' and i + 1 < len(text):
            # Skip escaped character
            i += 2
            continue
        
        if char == '{':
            count += 1
        elif char == '}':
            count -= 1
        
        if count == 0:
            return text[start_pos+1:i], i + 1
        i += 1
    return None, start_pos

def escape_underscores_in_text(text):
    """Escapes underscores in text if they are not already escaped."""
    return re.sub(r'(?<!\\)_', r'\_', text)

def process_text_commands(text):
    """Finds commands like \texttt{...} and escapes underscores inside their arguments."""
    commands = [
        'texttt', 'textbf', 'textit', 'textsf', 'textsc', 'textsl', 'emph', 
        'section', 'subsection', 'subsubsection', 'paragraph', 'subparagraph', 
        'caption', 'footnote', 'url', 'label', 'ref', 'cite', 'definition', 'algobox',
        'text', 'mathrm', 'mathbf', 'mathsf', 'mathtt', 'mathit'
    ]
    
    for cmd in commands:
        # Regex to find \cmd{
        pattern = re.compile(r'\\' + cmd + r'(?=\{)')
        pos = 0
        while True:
            match = pattern.search(text, pos)
            if not match:
                break
            
            start_brace = match.end()
            inner_text, end_brace = find_brace_content(text, start_brace)
            
            if inner_text is not None:
                escaped_inner = escape_underscores_in_text(inner_text)
                # Recursively process inner text for other commands
                escaped_inner = process_text_commands(escaped_inner)
                
                # Replace the old inner text with escaped one
                text = text[:start_brace+1] + escaped_inner + text[end_brace-1:]
                # Update pos to continue search after the current command
                pos = start_brace + 1 + len(escaped_inner) + 1
            else:
                pos = match.end()
    
    return text

def process_file_content(content):
    placeholders = []
    
    def get_placeholder(block_content, kind):
        idx = len(placeholders)
        placeholders.append(block_content)
        return f"@@@PLACEHOLDER_{kind}_{idx}@@@"

    # 1. Protect jupyter environments
    jupyter_envs = ['jupyterinput', 'jupyteroutput']
    for env in jupyter_envs:
        pattern = re.compile(r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}', re.DOTALL)
        content = pattern.sub(lambda m: get_placeholder(m.group(0), 'JUPYTER'), content)

    # 2. Protect math mode blocks
    # Named environments
    math_envs = [
        'equation', 'align', 'gather', 'multline', 'flalign', 'alignat', 'displaymath',
        'equation*', 'align*', 'gather*', 'multline*', 'flalign*', 'alignat*', 'displaymath*',
        'algorithmic' # Adding algorithmic as well, we'll process it like math (protect math inside it)
    ]
    for env in math_envs:
        pattern = re.compile(r'\\begin\{' + re.escape(env) + r'\}.*?\\end\{' + re.escape(env) + r'\}', re.DOTALL)
        content = pattern.sub(lambda m: get_placeholder(process_text_commands(m.group(0)), 'MATH'), content)

    # \[ ... \] and \( ... \)
    content = re.sub(r'\\\[.*?\\\]', lambda m: get_placeholder(process_text_commands(m.group(0)), 'MATH'), content, flags=re.DOTALL)
    content = re.sub(r'\\\(.*?\\\)', lambda m: get_placeholder(process_text_commands(m.group(0)), 'MATH'), content, flags=re.DOTALL)

    # $$ ... $$
    content = re.sub(r'\$\$.*?\$\$', lambda m: get_placeholder(process_text_commands(m.group(0)), 'MATH'), content, flags=re.DOTALL)
    
    # $ ... $ (avoiding placeholders)
    # We use a trick to avoid matching placeholders: replace placeholders with something that doesn't have $
    # But placeholders already don't have $.
    # The only issue is if the text contains $ inside a placeholder (it shouldn't, as we processed math already).
    content = re.sub(r'(?<!\\)\$.*?(?<!\\)\$', lambda m: get_placeholder(process_text_commands(m.group(0)), 'MATH'), content, flags=re.DOTALL)

    # 3. Escape all remaining underscores
    content = escape_underscores_in_text(content)

    # 4. Restore placeholders
    placeholder_pattern = re.compile(r'@@@PLACEHOLDER_([A-Z]+)_(\d+)@@@')
    
    # We might have nested placeholders if we were not careful, but we processed them in an order
    # that should minimize this. Let's do a loop to be sure.
    while placeholder_pattern.search(content):
        def restore(match):
            idx = int(match.group(2))
            return placeholders[idx]
        content = placeholder_pattern.sub(restore, content)

    return content

if __name__ == "__main__":
    target_dir = "documentation/elements_eng"
    files = [f for f in os.listdir(target_dir) if f.endswith(".tex")]
    
    for filename in files:
        filepath = os.path.join(target_dir, filename)
        print(f"Processing {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = process_file_content(content)
        
        if new_content != content:
            print(f"Updating {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            print(f"No changes for {filepath}")
