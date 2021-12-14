************************************************************************
file with basedata            : cm154_.bas
initial value random generator: 2043524650
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  71
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       27        9       27
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        1          3           5   8  11
   3        1          3          11  12  13
   4        1          3           6   7  11
   5        1          3           9  12  15
   6        1          2          10  13
   7        1          2          16  17
   8        1          1           9
   9        1          2          10  14
  10        1          1          17
  11        1          1          15
  12        1          2          14  16
  13        1          3          14  15  16
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
  2      1    10       7    8    1    5
  3      1     1       5    3    7    4
  4      1    10       4    9    3    4
  5      1     1       7    3    8    5
  6      1     2       6    7    3    2
  7      1     3       6    7    2    5
  8      1     1       1    7    3    5
  9      1     7       3    8    4    3
 10      1     3       7    7    7    6
 11      1     8       4    3    5    3
 12      1     6       1    9    7    9
 13      1     1       1    7    7    4
 14      1     3       3    3    5    3
 15      1     8       7   10    9    7
 16      1     1       8    1    9    5
 17      1     6       2    5    5    4
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   22   85   74
************************************************************************
