************************************************************************
file with basedata            : c1551_.bas
initial value random generator: 400527270
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  130
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       26       14       26
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   9  15
   3        3          3           6   7   8
   4        3          1          10
   5        3          1          12
   6        3          2          11  13
   7        3          2          12  16
   8        3          1          17
   9        3          1          16
  10        3          1          14
  11        3          3          12  14  16
  12        3          1          17
  13        3          1          17
  14        3          1          15
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     8       0    3    3    2
         2     9       0    2    2    2
         3    10       0    2    2    1
  3      1     1       8    0    5    2
         2     5       7    0    4    2
         3     6       7    0    3    2
  4      1     4       8    0    9    8
         2     6       5    0    8    8
         3     6       0    5    8    6
  5      1     6       0    6   10    3
         2     7       8    0    9    3
         3    10       6    0    9    3
  6      1     2       8    0    5    7
         2     7       6    0    3    7
         3     8       0    6    2    5
  7      1     6       0    4    8    9
         2     8       0    4    7    7
         3     8       3    0    6    4
  8      1     1       3    0    6    4
         2     1       0    6    5    6
         3     1       8    0    5    5
  9      1     5       7    0    3    9
         2     6       6    0    3    8
         3    10       6    0    2    8
 10      1     5       5    0    5    5
         2     5       0    3    5    5
         3     8       0    2    5    2
 11      1     2       3    0    6    6
         2     3       0    7    3    5
         3    10       0    7    3    4
 12      1     5       2    0    6    8
         2    10       0    8    3    7
         3    10       0    6    5    8
 13      1     3      10    0    5    5
         2     7      10    0    4    3
         3     8       0    3    2    3
 14      1     3       0    8    3    9
         2     5       8    0    3    7
         3     9       0    5    2    7
 15      1     4       5    0    7    9
         2     7       5    0    7    5
         3     8       4    0    5    2
 16      1     1       6    0    5    5
         2     6       0    9    4    5
         3    10       0    9    3    4
 17      1     7       0    6   10    2
         2     7       0    7    8    2
         3     8       8    0    6    1
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   22   89   87
************************************************************************
