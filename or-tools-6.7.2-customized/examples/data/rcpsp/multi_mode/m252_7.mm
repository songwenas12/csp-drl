************************************************************************
file with basedata            : cm252_.bas
initial value random generator: 1343957842
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  120
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       33        3       33
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        2          2           5  11
   3        2          3           6   8   9
   4        2          2           6  11
   5        2          3           6   8  17
   6        2          3           7  10  12
   7        2          2          14  15
   8        2          2          14  15
   9        2          3          11  12  16
  10        2          2          13  16
  11        2          2          13  17
  12        2          1          13
  13        2          1          15
  14        2          1          16
  15        2          1          18
  16        2          1          18
  17        2          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     7       0    7    8    4
         2     9       4    0    5    4
  3      1     5       2    0    3    6
         2     6       0    4    2    3
  4      1     1       0    8    7    8
         2     7       0    8    7    6
  5      1     8       0    7    6    9
         2     9       1    0    6    6
  6      1     3       0    6    7    2
         2     3       0    3    7    3
  7      1     6       0    5    7    8
         2    10       0    3    1    4
  8      1     2       0    9    8    9
         2     9       0    9    6    8
  9      1     2       0    3    4    6
         2    10       1    0    4    4
 10      1     6      10    0    8    9
         2    10       0    5    7    7
 11      1     2       7    0    7    5
         2     3       5    0    5    4
 12      1     1       7    0    4    5
         2     4       7    0    3    2
 13      1     4       8    0    7    8
         2     6       0    5    2    8
 14      1     3       0   10    8    6
         2     8       0    5    8    4
 15      1     4       7    0    8    8
         2     7       0    4    7    8
 16      1     6      10    0    7    2
         2    10       2    0    5    2
 17      1     8       8    0    5    6
         2     9       1    0    3    5
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   25   19   98   96
************************************************************************
