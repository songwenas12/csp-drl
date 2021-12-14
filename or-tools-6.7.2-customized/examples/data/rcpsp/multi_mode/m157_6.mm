************************************************************************
file with basedata            : cm157_.bas
initial value random generator: 1200063895
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
    1     16      0       50        0       50
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        1          3           5  11  17
   3        1          3           6   7   9
   4        1          1           6
   5        1          3           8   9  14
   6        1          3          11  12  17
   7        1          3          10  11  12
   8        1          2          10  12
   9        1          2          10  13
  10        1          1          15
  11        1          1          16
  12        1          2          13  15
  13        1          1          16
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
  2      1     4       0    3    1    3
  3      1     4       4    0    5    7
  4      1     6      10    0   10    8
  5      1     7       3    0    5    6
  6      1     2       0    5    4    5
  7      1     4       0    7    5    7
  8      1    10       0    2    8    2
  9      1     2       8    0    1    2
 10      1    10       0    7    4    8
 11      1     3       0    9    4   10
 12      1    10       5    0    3    6
 13      1    10       0    8    3   10
 14      1     3       7    0    7    1
 15      1     2       0    9    7    9
 16      1     9       4    0    7   10
 17      1    10       3    0    2    7
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   12   11   76  101
************************************************************************
