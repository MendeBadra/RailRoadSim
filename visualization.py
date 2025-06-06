import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

# Define the terminals and connections
TERMINALS = {
    'Амгалан': {
        'Амгалан Терминал': {'distance': 15, 'capacity': 12, 'station': 'Ам', 'region': 1},
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

# Create directed graph
G = nx.DiGraph()

main_nodes = ['Замын үүд', 'Амгалан', 'Улаанбаатар', 'Толгойт']
for node in main_nodes:
    G.add_node(node, node_type='city', size=1000, color='#1f78b4')

main_connections = [('Замын үүд', 'Амгалан'), ('Амгалан', 'Улаанбаатар'), ('Улаанбаатар', 'Толгойт')]
for u, v in main_connections:
    G.add_edge(u, v, weight=3, color='#333333', style='solid')

for city, terminals in TERMINALS.items():
    for terminal, data in terminals.items():
        G.add_node(terminal, node_type='terminal', size=100, 
                   color='#33a02c' if data['region'] == 1 else '#e31a1c',
                   capacity=data['capacity'], distance=data['distance'])
        G.add_edge(city, terminal, weight=1, color='#a6cee3', style='dashed')

# Layout
fig, ax = plt.subplots(figsize=(16, 10))
pos = {}
x_positions = np.linspace(0, 10, len(main_nodes))
for i, city in enumerate(main_nodes):
    pos[city] = (x_positions[i], 5)

for city in main_nodes:
    if city in TERMINALS:
        terminals = [t for t in G.neighbors(city) if G.nodes[t]['node_type'] == 'terminal']
        angle_step = np.pi / (len(terminals) + 1)
        start_angle = np.pi / 2 - angle_step * (len(terminals) / 2)
        for i, terminal in enumerate(terminals):
            angle = start_angle + i * angle_step
            distance_scale = G.nodes[terminal]['distance'] * 0.05
            dx = distance_scale * np.cos(angle)
            dy = distance_scale * np.sin(angle)
            pos[terminal] = (pos[city][0] + dx, pos[city][1] + dy)

node_colors = [G.nodes[n]['color'] for n in G.nodes()]
node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
edge_colors = [G.edges[e]['color'] for e in G.edges()]
edge_styles = [G.edges[e]['style'] for e in G.edges()]
edge_widths = [G.edges[e]['weight'] for e in G.edges()]

# Draw graph
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                       style=edge_styles, arrowsize=20, arrowstyle='-|>', node_size=node_sizes, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
nx.draw_networkx_labels(G, pos, labels={n: n for n in main_nodes},
                        font_size=10.2, font_weight='bold', font_family='sans-serif', ax=ax)
nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes() if n not in main_nodes},
                        font_size=8.5, font_family='sans-serif', ax=ax)

plt.title("Чингэлэг тээврийн сүлжээ: Замын үүд-Амгалан-Улаанбаатар-Толгойт\nГалт тэрэгний хөдөлгөөн", 
          fontsize=18, pad=25, fontweight='bold')

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='', markerfacecolor='#1f78b4', markersize=15),
    plt.Line2D([0], [0], marker='o', color='w', label='Бүс 1 терминал', markerfacecolor='#33a02c', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Бүс 2 терминал', markerfacecolor='#e31a1c', markersize=10),
    plt.Line2D([0], [0], color='#333333', lw=2, label='Гол тээврийн зам'),
    plt.Line2D([0], [0], color='#a6cee3', lw=2, linestyle='--', label='Терминалын зам')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Train animation setup
train_path = ['Замын үүд', 'Амгалан', 'Улаанбаатар', 'Толгойт']
path_positions = [pos[city] for city in train_path]
train_length, train_height = 0.7, 0.03
train_rect = Rectangle((0, 0), train_length, train_height, angle=0, color='red', alpha=0.9)
ax.add_patch(train_rect)
train_rect.set_visible(False)

# Dots for terminals
dots = []
dot_speeds = []
dot_targets = []
dot_colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'pink']
active_dots = []
stop_frames = {1: 30, 2: 30, 3: 30}
  # Index of stop in train_path: pause duration (frames)
current_state = {'frame': 0, 'train_index': 0, 'pause': 0, 'dots_done': True}

def create_dots(city_index):
    city = train_path[city_index]
    terminals = [t for t in G.neighbors(city) if G.nodes[t]['node_type'] == 'terminal']
    dots.clear()
    dot_targets.clear()
    dot_speeds.clear()
    active_dots.clear()
    for i, term in enumerate(terminals):
        dot = Circle(pos[city], radius=0.05, color=dot_colors[i % len(dot_colors)])
        ax.add_patch(dot)
        dots.append(dot)
        dot_targets.append(pos[term])
        dot_speeds.append(0.1)
        active_dots.append(True)

def update(frame):
    cs = current_state

    # 1. Handle dot animation or pause after it
    if not cs['dots_done']:
        all_done = True
        for i, dot in enumerate(dots):
            if not active_dots[i]:
                continue
            cur_x, cur_y = dot.center
            tgt_x, tgt_y = dot_targets[i]
            dx, dy = tgt_x - cur_x, tgt_y - cur_y
            dist = np.hypot(dx, dy)
            if dist < dot_speeds[i]:
                dot.center = (tgt_x, tgt_y)
                active_dots[i] = False
            else:
                dot.center = (cur_x + dx * dot_speeds[i], cur_y + dy * dot_speeds[i])
                all_done = False
        if all_done:
            cs['dots_done'] = True
            cs['pause'] = stop_frames.get(cs['train_index'], 10)
        return [train_rect] + dots

    if cs['pause'] > 0:
        cs['pause'] -= 1
        return [train_rect] + dots

    # 2. Check if train finished entire path
    if cs['train_index'] >= len(train_path) - 1:
        return [train_rect]

    # 3. Move train to next station
    cs['frame'] += 1
    segment_frames = 30
    seg_index = cs['train_index']
    progress = cs['frame'] / segment_frames

    start_pos = path_positions[seg_index]
    end_pos = path_positions[seg_index + 1]
    x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
    y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
    angle = np.degrees(np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0]))
    train_rect.set_xy((x - train_length / 2, y - train_height / 2))
    train_rect.angle = angle
    train_rect.set_visible(True)

    # 4. Arrived at next station
    if progress >= 1.0:
        cs['train_index'] += 1
        cs['frame'] = 0

        if cs['train_index'] in stop_frames:
            create_dots(cs['train_index'])
            cs['dots_done'] = False

    return [train_rect] + dots

# Animate
ani = FuncAnimation(fig, update, frames=300, interval=100, blit=True)

plt.tight_layout()
plt.axis('off')
plt.show()
