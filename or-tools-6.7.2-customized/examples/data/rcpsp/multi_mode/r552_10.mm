************************************************************************
file with basedata            : cr552_.bas
initial value random generator: 623789505
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  131
RESOURCES
  - renewable                 :  5   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       31       14       31
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   7
   3        3          3           8  11  14
   4        3          2           7   8
   5        3          3           6   8  10
   6        3          1           9
   7        3          2           9  11
   8        3          2          15  17
   9        3          3          12  14  16
  10        3          3          11  14  16
  11        3          1          12
  12        3          2          13  17
  13        3          1          15
  14        3          2          15  17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  R 4  R 5  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0    0    0    0
  2      1     7       0    0    0    9    3    7   10
         2     8       0    3    6    8    0    7    8
         3     9       6    0    0    4    0    6    6
  3      1     7      10    3    0    0    0    5   10
         2     8       0    0    7    0    0    4    8
         3    10       9    3    5    2    0    4    6
  4      1     1       6    8    0    8    0    7    9
         2     4       0    0    8    7    6    6    6
         3     8       0    5    7    0    0    6    2
  5      1     3      10    0    0    0    0    5    9
         2     3       0    3    0    0    2    8    7
         3     4       0    1    0   10    0    2    4
  6      1     4       0    0    1    0    9    9    9
         2     5       6    2    0    5    0    9    9
         3     8       2    0    0    0    0    9    8
  7      1     9       0    8    8    0    0    8    6
         2     9       5    0    3    0    4    7    9
         3     9       4    8    0    0    3    6    7
  8      1     1       8    0    9    0    8    5    6
         2     6       0    0    0    2    0    4    5
         3     9       6    0    0    0    6    2    5
  9      1     2       8    0    6    8    0    4   10
         2     5       7    0    5    7    0    3    9
         3     7       7    0    0    7    0    3    9
 10      1     2       3    0    8    0    7    6    4
         2     6       3    0    6    0    7    5    3
         3     7       2    5    1    0    7    3    3
 11      1     3       0    7    0    0    9    9    8
         2     4       0    0    0    0    8    8    5
         3    10       0    0    6    7    0    8    5
 12      1     2       5    0    0   10    2   10    4
         2     8       4    0    0    0    0   10    3
         3     9       3    0    8    5    0    9    1
 13      1     3       9    5    5    6    3    7    3
         2     3       8    4    6    2    5    8    2
         3     9       7    3    4    0    0    4    2
 14      1     2       1    0    0    4    8    9    7
         2     6       0    8    0    0    0    8    4
         3     7       0    7    3    0    0    6    3
 15      1     7       5    0    0    6    0    6    7
         2     8       0    0    0    6    0    6    7
         3    10       0    5    9    0    2    1    7
 16      1     5      10    0    0    3    5    8    8
         2     7       0    7    7    0    2    5    5
         3     9       9    0    0    1    1    4    5
 17      1     2       4    9    5    0    0    7    5
         2     4       0    9    4    0    7    6    5
         3     6       0    8    3    4    5    4    3
 18      1     0       0    0    0    0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  R 4  R 5  N 1  N 2
   23   22   26   19   28  106  107
************************************************************************
