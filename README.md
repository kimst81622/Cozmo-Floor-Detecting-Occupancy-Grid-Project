# Occupancy Grid via Floor Detection on Cozmo

Authors are Steven Kim and Hal Rockwell.

The main code is in `grid.py` and `classifiers.py`. The FSM
`PatchCollect.fsm` collects patches from a new surface and saves them,
while `GridTest.fsm` gives a demo of the occupancy grid, letting you
drive Cozmo around and see it update. Finally, a saved collection
of patches in stored in `floors/`, of Hal Rockwell's wood floor,
for quick use in the demo (as well as a subsampled version that
runs more efficiently).

A more thorough description, a video demo, and slides can be found at
[this webpage](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15494-s20/users/hrockwel/www/).
