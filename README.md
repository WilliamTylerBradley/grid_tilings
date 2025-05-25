# grid_tilings
Tilings from multigrids

Create a multigrid and use it to create a tiling pattern.
```
import grid_tilings

mg = grid_tilings.MultiGrid(grid_count=5,
                            grid_bounds=[-5, 5],
                            rotation=0,
                            base_point = [0, 0],
                            offsets=[.2, .2, .2, .2, .2])
mg.create_tiles()
```

Save a zoomed in image of the tiling so no background is seen.
```
mg.save_image('image_graph.png')
```

See one of the lines.
```
l = mg.grids[10]
fig, ax = plt.subplots()
l.draw_line(ax)
plt.savefig('line_graph.png')
plt.close()
```

See one of the tiles.
```
t = list(mg.tiles.values())[0]
fig, ax = plt.subplots()
t.draw_tile(ax)
plt.savefig('tile_graph.png')
plt.close()
```

See one of the tiles with it's lines from the multigrid.
```
fig, ax = plt.subplots()
t.draw_line_angles(ax)
plt.savefig('line_angles_graph.png')
plt.close()
```

See the grids from the multigrid.
```
fig, ax = plt.subplots()
mg.draw_grids(ax)
plt.savefig('grids_graph.png')
plt.close()
```

See the k values for one of the grids.
```
fig, ax = plt.subplots()
mg.draw_k(ax, 0)
plt.savefig('k_graph.png')
plt.close()
```

See the offsets.
```
fig, ax = plt.subplots()
mg.draw_offsets(ax)
plt.savefig('offsets_graph.png')
plt.close()
```

See all of the tiles.
```
fig, ax = plt.subplots()
mg.draw_all_tiles(ax)
plt.savefig('all_tiles_1.png')
plt.close()
```

Remove tiles that aren't connected to the center.
```
mg.remove_unconnected_tiles()

fig, ax = plt.subplots()
mg.draw_all_tiles(ax)
plt.savefig('all_tiles_2.png')
plt.close()
```