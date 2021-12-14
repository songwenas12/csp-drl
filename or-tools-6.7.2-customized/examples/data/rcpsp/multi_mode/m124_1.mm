************************************************************************
file with basedata            : cm124_.bas
initial value random generator: 28366
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  96
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       46        8       46
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        1          2           5   6
   3        1          3           5   7  12
   4        1          2          14  15
   5        1          3          14  15  16
   6        1          3           7   8  15
   7        1          2           9  10
   8        1          3           9  10  12
   9        1          1          13
  10        1          2          11  13
  11        1          2          16  17
  12        1          1          17
  13        1          2          14  16
  14        1          1          17
  15        1          1          18
  16        1          1          18
  17        1          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     7       7    1    0    9
  3      1     5       6    9    6    0
  4      1     8       7    6    0    9
  5      1     7       9    5    0    4
  6      1     5       3    9    6    0
  7      1     4       3    5    0    4
  8      1     1       9    3    0    5
  9      1    10       7   10    6    0
 10      1     8       6    1    3    0
 11      1     9      10    3    4    0
 12      1     1       3    7    9    0
 13      1    10       4    7    0    5
 14      1     6       1    2    0    5
 15      1     4       7    6    8    0
 16      1     7       6    5    2    0
 17      1     4       3    4    7    0
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   21   20   51   41
************************************************************************
