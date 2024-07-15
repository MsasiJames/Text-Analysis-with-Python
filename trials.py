import ast
import graphviz

def create_syntax_tree(sentence):
    """
    Automatically creates a syntax tree from the given sentence.
    
    Args:
        sentence (str): English sentence.
        
    Returns:
        ast.Module: The root node of the syntax tree.
    """
    code = f"print('{sentence}')"
    return ast.parse(code)

def draw_syntax_tree(root_node):
    """
    Draws the syntax tree graphically using Graphviz.
    
    Args:
        root_node (ast.Module): The root node of the syntax tree.
    """
    dot = graphviz.Digraph()

    def traverse(node, parent_name=None):
        name = str(node.__class__.__name__)
        if isinstance(node, ast.AST):
            name += f" ({type(node).__name__})"
        current_node_name = f"{id(node)}_{name}"
        dot.node(current_node_name, name)
        if parent_name:
            dot.edge(parent_name, current_node_name)
        for child_name, child_node in ast.iter_fields(node):
            if isinstance(child_node, list):
                for child in child_node:
                    traverse(child, current_node_name)
            elif isinstance(child_node, ast.AST):
                traverse(child_node, current_node_name)

    traverse(root_node)
    dot.render('syntax_tree', format='png', cleanup=True)

# Example usage:
sentence = "The big black dog barked at the white cat and chased away"
tree = create_syntax_tree(sentence)
draw_syntax_tree(tree)
