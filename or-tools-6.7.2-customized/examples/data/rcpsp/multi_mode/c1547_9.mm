************************************************************************
file with basedata            : c1547_.bas
initial value random generator: 78477998
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  122
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       20       15       20
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           6  16
   3        3          1           6
   4        3          2           5  15
   5        3          1           8
   6        3          3           7  11  13
   7        3          2           9  15
   8        3          3           9  10  11
   9        3          1          17
  10        3          2          12  14
  11        3          1          17
  12        3          1          16
  13        3          1          15
  14        3          1          16
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       8    8    7    9
         2     9       6    6    7    9
         3    10       3    6    3    9
  3      1     1       6   10    7    6
         2     1       6   10    6    7
         3     8       5   10    6    3
  4      1     1       8    6    7    4
         2     3       8    6    5    4
         3     8       7    5    3    4
  5      1     5       1    7    9    8
         2     6       1    6    8    6
         3    10       1    4    6    5
  6      1     4       8    6    8    8
         2     6       8    6    6    5
         3     6       7    6    7    5
  7      1     1       8    8    3    8
         2     1       6    8    3    9
         3     7       5    7    2    3
  8      1     1       6    8    9    9
         2     3       6    8    9    7
         3     6       6    4    9    6
  9      1     3       3    9    6    5
         2     7       2    9    6    4
         3    10       2    9    4    3
 10      1     2       8   10    9    6
         2     4       6    8    6    4
         3     5       3    8    6    2
 11      1     5       3    3    9    8
         2     6       2    2    7    8
         3     8       2    2    4    8
 12      1     3       6    3    5   10
         2     3       6    3    6    9
         3     5       5    3    2    8
 13      1     5       9    7    6    5
         2     5       8    7    7    7
         3     6       7    4    5    5
 14      1     3       8    6    4   10
         2     3       8    5    5    9
         3     8       4    3    4    9
 15      1     2       7    9    6    4
         2     6       6    7    5    3
         3     8       4    7    3    2
 16      1     3       5    7    5   10
         2     6       5    7    5    5
         3     7       4    6    4    4
 17      1     8       3    7    9    6
         2    10       2    7    8    4
         3    10       3    7    7    6
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   24   24   93  100
************************************************************************
