************************************************************************
file with basedata            : cm135_.bas
initial value random generator: 1644312364
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  93
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       34        9       34
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        1          3           6  10  13
   3        1          3           5   8  14
   4        1          2           9  10
   5        1          2           7   9
   6        1          2           7   9
   7        1          2          11  16
   8        1          3          12  13  15
   9        1          3          11  12  15
  10        1          1          12
  11        1          1          17
  12        1          2          16  17
  13        1          1          17
  14        1          2          15  16
  15        1          1          18
  16        1          1          18
  17        1          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     5       0    7    8    4
  3      1     4       3    0    8    5
  4      1     7       7    0   10    7
  5      1     8       3    0    4    5
  6      1     2       0    6    9    6
  7      1     8       0    3    7    4
  8      1     6       4    0    9    2
  9      1     2       0    7    7    7
 10      1     2       7    0    3    6
 11      1    10       0    2    8    6
 12      1     8       0    4    8    7
 13      1    10       8    0    6    3
 14      1     5       4    0   10    9
 15      1     8       6    0    2    9
 16      1     4       0   10    3    2
 17      1     4       8    0    2    7
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   12  104   89
************************************************************************
