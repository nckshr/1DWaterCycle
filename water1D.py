import numpy as np
from numpy import random as nprandom
from graphics import *
import sys, signal, time, random

def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)

def runSimulation(dt,dx,g,heights,momentums,num_steps):
  
  num_cells = heights.shape[0]
  h = heights
  hv = momentums
  #print(h_log[0])
  #return h_log
  for t in range(0,num_steps):
    #Half step 
    hm = (h[0:(num_cells-1)] + h[1:num_cells]) / 2.0
    hvm = (hv[0:(num_cells-1)] + hv[1:num_cells]) / 2.0
    dhvm_dx = (hv[1:num_cells] - hv[0:(num_cells-1)]) / dx
    dhm_dt = - dhvm_dx
    hm += dhm_dt * (dt/2.0)
    dhvm2_dx = ((hv[1:num_cells]**2 / h[1:num_cells]) - (hv[0:(num_cells-1)]**2/h[0:(num_cells-1)])) / dx
    dgh2_dx = 0.5 * g * (h[1:num_cells]**2 - h[0:(num_cells-1)]**2) / dx
    dhvm_dt = -dhvm2_dx - dgh2_dx
    hvm += dhvm_dt * (dt/2)
    
    #Full step
    dhv_dx = (hvm[1:(num_cells-1)] - hvm[0:(num_cells-2)]) / dx
    dh_dt = - dhv_dx
    h_tp1 = np.hstack((h[1] + dh_dt[0] * dt, h[1:(num_cells-1)] + dh_dt * dt, h[(num_cells-2)] + dh_dt[-1] * dt))
    dhv2_dx = ((hvm[1:(num_cells-1)]**2 / hm[1:(num_cells-1)]) - (hvm[0:(num_cells-2)]**2/hm[0:(num_cells-2)])) / dx
    dgh2_dx = 0.5 * g * (hm[1:(num_cells-1)]**2 - hm[0:(num_cells-2)]**2) / dx
    dhv_dt = -dhv2_dx - dgh2_dx
    hv_tp1 = np.hstack(( -(hv[1] + dhv_dt[0] * dt), hv[1:(num_cells-1)] + dhv_dt * dt, -(hv[(num_cells-2)] + dhv_dt[-1] * dt) ))
    #momentum dampening
    hv_tp1 *= 0.99
    h = h_tp1
    hv = hv_tp1
  return h, hv

  return h,hv
def initDrawing(win,heights):
  num_cells = heights.shape[0]
  cell_size = win.width/num_cells
  columns = []
  for i in range(heights.shape[0]):
    bottom_left = Point(i*cell_size,win.height)
    bottom_right = Point(i*cell_size+cell_size,win.height)
    top_left = Point(i*cell_size,win.height-heights[i])
    top_right = Point(i*cell_size+cell_size,win.height-heights[i])
    verts = [bottom_left,top_left,top_right,bottom_right]
    column = Polygon(verts)
    column.setFill("blue")
    column.draw(win)
    columns.append(column)
  return columns
def drawWater(win,polygons, heights,scale):
  num_cells = heights.shape[0]
  cell_size = win.width/num_cells
  #convert to render space
  heights = heights/ 16.0 * win.height
  for i in range(heights.shape[0]):
    polygons[i].undraw()
    bottom_left = Point(i*cell_size,win.height)
    bottom_right = Point(i*cell_size+cell_size,win.height)
    top_left = Point(i*cell_size,win.height-heights[i])
    top_right = Point(i*cell_size+cell_size,win.height-heights[i])
    polygons[i]= Polygon([bottom_left,top_left,top_right,bottom_right])
    polygons[i].setFill("blue")
    polygons[i].draw(win)
  win.update()

def main():
  
  rain_chance = 0.1 
  signal.signal(signal.SIGINT, signal_handler)
  win = GraphWin("Water1D",600,600, autoflush=False)
  dt = 0.02
  steps_per_draw = 1
  dx = 2.0/16.0
  g = 1.0
  scale =8
  total_size = 2
  positions = np.linspace(0,2,int(total_size/dx))
  h = (1.0/np.sqrt(0.5*2.0*np.pi)) * np.exp(-0.5*(np.power((positions-1.0)/0.5,2))) * 2
  #h = np.ones(positions.shape[0]) * 1.5
  num_cells = h.shape[0]
  cell_size = win.width/num_cells
  hv = np.zeros_like(h)
  
  polys = initDrawing(win,h)
  print(h)
  drawWater(win,polys,h,scale)
  #hv = [np.zeros_like(x)]*num_steps
  #i = 0
  paused = True
  drops_per_sec = 0.2
  last_start = time.time()
  drops = []
  drop_speed = 7
  time_till_next_drop = min(5,nprandom.exponential(scale=1.0/drops_per_sec,size=1)[0])
  drain = False
  while True:
    k = win.checkKey()
    if k == 'p':
      paused = not paused
    elif k == 'r':
      h = (1.0/np.sqrt(0.5*2.0*np.pi)) * np.exp(-0.5*(np.power((positions-1.0)/0.5,2))) * 2
      hv = np.zeros_like(h)
    elif k.isdigit():
      c = int(k)
      cprev = max(c-1,0)
      cnext = min(c+1,num_cells-1)
      hv[cprev] = hv[c] - 0.5
      hv[cnext] = hv[c] + 0.5
      h[c] = h[c]+0.5
    frame_start = time.time()
    time_elapsed = frame_start - last_start
    last_start = frame_start
    if not paused:
      time_till_next_drop = time_till_next_drop - time_elapsed
      #print(f"next drop in {time_till_next_drop} sec")
      if time_till_next_drop <= 0:
        dropcol = random.randint(0,15)
        drops.append({'circle':Circle(Point(dropcol*cell_size + cell_size/2.0,16),cell_size/2.0),'col':dropcol,'pos':16})
        time_till_next_drop =  min(5,nprandom.exponential(scale=1.0/drops_per_sec,size=1)[0])
        #print("adding drop!")
      #step water sim
      h, hv = runSimulation(dt,dx,g,h,hv,steps_per_draw)
      #convert to grid space
      heights_grid = np.floor((h * scale / 16.0) * 16.0 + 0.5)
      next_drops = []
      for i in range(len(drops)):
        drops[i]['circle'].undraw()
        drops[i]['pos'] = drops[i]['pos'] - time_elapsed * drop_speed;
        drops[i]['circle'].__init__(Point(drops[i]['col'] * cell_size + cell_size/2.0,win.height - np.floor(drops[i]['pos'])/16.0 * win.height),cell_size/2.0)
        if drops[i]['pos'] <= heights_grid[i]:
          cprev = max(drops[i]['col']-1,0)
          cnext = min(drops[i]['col']+1,num_cells-1)
          hv[cprev] = hv[drops[i]['col']] - 0.6
          hv[cnext] = hv[drops[i]['col']] + 0.6
          h[drops[i]['col']] = h[drops[i]['col']]+0.6
        else:
          drops[i]['circle'].setFill("blue")
          drops[i]['circle'].draw(win)
          next_drops.append(drops[i])
      
      drops = next_drops
      if np.min(heights_grid) > 10:
        drain = True
      if drain:
        h[14] = h[14] - 3 * time_elapsed
        if(heights_grid[14] < 5):
          drain = False
      #end = time.time()
      #print(f"simulation took {end - start}")
      #start = time.time()
      drawWater(win,polys,heights_grid,scale)
      #end = time.time()
      #print(f"drawing took {end - start}")
      #print(f"iteration: {i}")
    time.sleep(0.0016)
    #i = i+1


if __name__ == "__main__":
    main()