import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

# Define the terminals and connections
TERMINALS = {
    'Амгалан': {
        'Амгалан': {'distance': 15, 'capacity': 12, 'station': 'Ам', 'region': 1},
    },
    'Улаанбаатар': {
        'УБ МЧ': {'distance': 8, 'capacity': 33, 'station': 'УБ', 'region': 1},
        'Туушин': {'distance': 20, 'capacity': 8, 'station': 'УБ', 'region': 1},
        'Монгол экс': {'distance': 20, 'capacity': 10, 'station': 'УБ', 'region': 1},
        'материалын импекс': {'distance': 40, 'capacity': 23, 'station': 'УБ', 'region': 2},
        'Прогресс': {'distance': 60, 'capacity': 13, 'station': 'УБ', 'region': 1},
        'Эрин': {'distance': 60, 'capacity': 7, 'station': 'УБ', 'region': 2},
        'Техник импорт': {'distance': 40, 'capacity': 12, 'station': 'УБ', 'region': 2}, 
    },
    'Толгойт': {
        'Интердэсишн': {'distance': 20, 'capacity': 8, 'station': 'То', 'region': 1},
        'Монгол транс': {'distance': 10, 'capacity': 15, 'station': 'То', 'region': 1},
        'Толгойт МЧ': {'distance': 5, 'capacity': 10, 'station': 'То', 'region': 1},
        'Номин Трэйдинг': {'distance': 20, 'capacity': 8, 'station': 'То', 'region': 2},
    }
}

# Create a directed graph
G = nx.DiGraph()

# Add main nodes (cities)
main_nodes = ['Эрээн', 'Амгалан', 'Улаанбаатар', 'Толгойт']
for node in main_nodes:
    G.add_node(node, node_type='city', size=1000, color='#1f78b4')  # Blue for cities

# Add connections between main nodes
main_connections = [('Эрээн', 'Амгалан'), 
                   ('Амгалан', 'Улаанбаатар'), 
                   ('Улаанбаатар', 'Толгойт')]
for u, v in main_connections:
    G.add_edge(u, v, weight=3, color='#333333', style='solid')

# Add terminal nodes and connections
for city, terminals in TERMINALS.items():
    for terminal, data in terminals.items():
        G.add_node(terminal, 
                  node_type='terminal',
                  size=100, 
                  color='#33a02c' if data['region'] == 1 else '#e31a1c',  # Green/Red
                  capacity=data['capacity'],
                  distance=data['distance'])
        G.add_edge(city, terminal, weight=1, color='#a6cee3', style='dashed')

# Set up the plot with better layout control
fig, ax = plt.subplots(figsize=(16, 10))

# Create a structured layout
pos = {}
# Main nodes in a horizontal line
x_positions = np.linspace(0, 10, len(main_nodes))
for i, city in enumerate(main_nodes):
    pos[city] = (x_positions[i], 5)  # Cities at y=5

# Position terminals in arcs above their cities
for city in main_nodes:
    if city in TERMINALS:
        terminals = [t for t in G.neighbors(city) if G.nodes[t]['node_type'] == 'terminal']
        angle_step = np.pi / (len(terminals) + 1)
        start_angle = np.pi/2 - angle_step * (len(terminals)/2)

        for i, terminal in enumerate(terminals):
          angle = start_angle + i * angle_step
          distance_scale = G.nodes[terminal]['distance'] * 0.05  # Adjust 0.05 as needed
          dx = distance_scale * np.cos(angle)
          dy = distance_scale * np.sin(angle)
          pos[terminal] = (pos[city][0] + dx, pos[city][1] + dy)

# Get node attributes
node_colors = [G.nodes[n]['color'] for n in G.nodes()]
node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
edge_colors = [G.edges[e]['color'] for e in G.edges()]
edge_styles = [G.edges[e]['style'] for e in G.edges()]
edge_widths = [G.edges[e]['weight'] for e in G.edges()]

# Draw all edges at once with proper attributes
nx.draw_networkx_edges(G, pos, 
                      width=edge_widths,
                      edge_color=edge_colors,
                      style=edge_styles,
                      arrowsize=20,
                      arrowstyle='-|>',
                      node_size=node_sizes,
                      ax=ax)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

# Draw labels with different styles for cities and terminals
city_labels = {n: n for n in main_nodes}
terminal_labels = {n: n for n in G.nodes() if n not in main_nodes}

nx.draw_networkx_labels(G, pos, labels=city_labels, 
                       font_size=10.2, font_weight='bold', font_family='sans-serif', ax=ax)
nx.draw_networkx_labels(G, pos, labels=terminal_labels, 
                       font_size=8.5, font_family='sans-serif', ax=ax)

# Add informative title with larger font
plt.title("Чингэлэг тээврийн сүлжээ: Эрээн-Амгалан-Улаанбаатар-Толгойт\n"
         "Галт тэрэгний хөдөлгөөн", 
         fontsize=18, pad=25, fontweight='bold')

# Add legend with adjusted font size
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Гол хот', 
              markerfacecolor='#1f78b4', markersize=15),
    plt.Line2D([0], [0], marker='o', color='w', label='Бүс 1 терминал', 
              markerfacecolor='#33a02c', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Бүс 2 терминал', 
              markerfacecolor='#e31a1c', markersize=10),
    plt.Line2D([0], [0], color='#333333', lw=2, label='Гол тээврийн зам'),
    plt.Line2D([0], [0], color='#a6cee3', lw=2, linestyle='--', label='Терминалын холбоос')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Create the train path (main route)
train_path = ['Эрээн', 'Амгалан', 'Улаанбаатар', 'Толгойт']
path_positions = [pos[city] for city in train_path]

# Create a triangle to represent the train - adjusted size
triangle_size = 0.1  # Increased from 0.3 to make it more visible
triangle = Polygon([[0, 0], 
                    [-triangle_size, -triangle_size*1.5], 
                    [triangle_size, -triangle_size*1.5]], 
                   closed=True, color='red', alpha=0.8)
ax.add_patch(triangle)
triangle.set_visible(False)

# Animation function
def update(frame):
    # Calculate progress along the entire path
    total_frames = 100
    segments = len(train_path) - 1
    segment_length = total_frames // segments
    current_segment = frame // segment_length
    progress = (frame % segment_length) / segment_length
    
    # Ensure we don't go out of bounds
    if current_segment >= segments:
        current_segment = segments - 1
        progress = 1.0
    
    # Get current segment start and end points
    start_pos = path_positions[current_segment]
    end_pos = path_positions[current_segment + 1]
    
    # Interpolate position
    x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
    y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
    
    # Update triangle position with new size
    triangle.set_xy([[x, y], 
                    [x - triangle_size, y - triangle_size*1.5], 
                    [x + triangle_size, y - triangle_size*1.5]])
    triangle.set_visible(True)
    
    # Simple rotation effect based on direction
    if current_segment < segments - 1:
        next_pos = path_positions[current_segment + 1]
        dx = next_pos[0] - end_pos[0]
        dy = next_pos[1] - end_pos[1]
        angle = np.arctan2(dy, dx)
    else:
        angle = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
    
    # Apply rotation
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                          [np.sin(angle), np.cos(angle)]])
    original = np.array([[x, y], 
                        [x - triangle_size, y - triangle_size*1.5], 
                        [x + triangle_size, y - triangle_size*1.5]])
    center = original.mean(axis=0)
    rotated = np.dot(original - center, rot_matrix) + center
    triangle.set_xy(rotated)
    
    return triangle,

# Create animation with adjusted parameters
ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

plt.tight_layout()
plt.axis('off')
plt.show()