************************************************************************
file with basedata            : cr453_.bas
initial value random generator: 1696348366
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  130
RESOURCES
  - renewable                 :  4   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       16       10       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   9  14
   3        3          3           8  11  14
   4        3          3           6  12  15
   5        3          3           7  11  12
   6        3          2          16  17
   7        3          2           8  10
   8        3          2          13  16
   9        3          3          10  11  13
  10        3          1          15
  11        3          1          17
  12        3          2          13  16
  13        3          1          17
  14        3          1          15
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  R 4  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0    0    0
  2      1     1       5    5    9   10    9    2
         2     2       5    4    7    6    7    2
         3     8       5    3    4    5    7    1
  3      1     1       2    8    5    7    6    9
         2     1       3    8    5    9    7    6
         3     3       1    4    3    2    4    2
  4      1     1       8   10    7    7    8    8
         2     5       8    9    4    4    8    8
         3     7       8    9    4    4    7    7
  5      1     4      10    5    9    7    7    5
         2     5      10    4    8    7    6    4
         3     6       9    4    7    6    5    4
  6      1     1       9   10    4    6   10    5
         2     5       9   10    3    5    9    5
         3     9       9   10    3    3    9    4
  7      1     2       5    3    3    9    7    6
         2     2       5    4    5    5    7    7
         3     3       4    2    2    1    2    4
  8      1     2       8    8    7    7    3    9
         2     3       8    6    6    5    3    9
         3    10       7    5    6    4    3    8
  9      1     2       7    8    9    8    9    8
         2     9       7    4    7    8    9    7
         3    10       6    1    6    8    8    7
 10      1     2       6    9    3    6    9    5
         2     6       6    9    3    5    7    4
         3    10       6    9    3    4    5    2
 11      1     7       7    8    8    8    7    6
         2     8       3    6    8    4    2    6
         3     8       4    7    8    6    2    5
 12      1     6       3    6    3    2    6    9
         2     9       3    6    2    1    5    8
         3    10       2    5    1    1    4    8
 13      1     1       7    3    6   10    5    9
         2     4       6    2    4   10    3    6
         3    10       3    2    4   10    2    4
 14      1     4       6    4    5    9    5    5
         2     8       4    2    5    7    5    5
         3    10       3    1    4    5    4    2
 15      1     1      10    6    6    9    8    9
         2     2       5    6    5    9    8    7
         3    10       5    2    3    8    7    4
 16      1     5       8    7    5    5   10    5
         2     5       9    8    6    5    9    4
         3     6       2    2    3    4    8    4
 17      1     3       6    2   10    5    3    6
         2     8       4    2    7    4    3    5
         3    10       3    1    7    1    3    2
 18      1     0       0    0    0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  R 4  N 1  N 2
   15   15   13   15  105   97
************************************************************************
