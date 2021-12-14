************************************************************************
file with basedata            : cr332_.bas
initial value random generator: 1463674396
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  123
RESOURCES
  - renewable                 :  3   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       19        9       19
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6  11  15
   3        3          3          11  13  15
   4        3          3           5   7  13
   5        3          2           6   8
   6        3          2          14  17
   7        3          2          10  14
   8        3          2           9  10
   9        3          2          11  12
  10        3          2          12  16
  11        3          2          14  17
  12        3          2          15  17
  13        3          1          16
  14        3          1          16
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0    0
  2      1     5       5    7    9    0    4
         2     6       3    7    6    9    0
         3    10       2    6    5    0    4
  3      1     1       8    6   10    0    6
         2     7       7    6    8    6    0
         3    10       6    4    7    0    6
  4      1     1       8    9    4    0    9
         2     4       8    8    2    9    0
         3     6       7    5    1    6    0
  5      1     6       9   10   10    9    0
         2     7       8    8   10    0    8
         3    10       8    6   10    9    0
  6      1     3      10    6    9    0    2
         2     4       8    6    6    0    2
         3     5       6    6    1    5    0
  7      1     2       7    7   10    0    4
         2     4       7    6   10    0    3
         3     9       7    5   10    7    0
  8      1     4       7    7    6    0    1
         2     5       5    4    4    0    1
         3    10       1    4    3    0    1
  9      1     2       8    5    8    0    4
         2     7       8    3    7    5    0
         3     8       7    3    7    0    4
 10      1     3       5    9   10    0    4
         2     7       5    7    9    7    0
         3     8       4    5    8    6    0
 11      1     4       7    4    7    0    1
         2     8       7    3    7    5    0
         3     9       4    1    5    4    0
 12      1     1       9    9    9    5    0
         2     6       5    8    6    4    0
         3     9       3    6    6    0    6
 13      1     5       3    4    3    7    0
         2     6       3    2    2    0    2
         3     6       1    2    2    4    0
 14      1     1      10    3    4    0    6
         2     8       7    3    3    6    0
         3    10       4    2    2    5    0
 15      1     3       4    6    4    0    4
         2     6       3    3    4    5    0
         3     6       2    1    3    0    3
 16      1     1      10    7   10    0    4
         2     4       9    5    4    0    4
         3     5       9    4    1    0    3
 17      1     2       2    4   10    8    0
         2     2       3    5    7    7    0
         3     2       1    5    6    0    5
 18      1     0       0    0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  N 1  N 2
   24   28   32   93   70
************************************************************************
