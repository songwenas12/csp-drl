************************************************************************
file with basedata            : cm431_.bas
initial value random generator: 1815822037
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  135
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       13        5       13
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        4          3           6   7  16
   3        4          3           8  11  13
   4        4          2           5   7
   5        4          2           8  12
   6        4          3           9  10  11
   7        4          3          10  11  12
   8        4          3          10  14  17
   9        4          2          12  17
  10        4          1          15
  11        4          1          17
  12        4          1          13
  13        4          1          15
  14        4          2          15  16
  15        4          1          18
  16        4          1          18
  17        4          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       8    9    0    2
         2     3       8    8    7    0
         3     4       8    6    0    2
         4     6       7    5    0    2
  3      1     1      10    5    0    6
         2     1      10    5    9    0
         3     2       8    4    4    0
         4     5       6    1    0    7
  4      1     2       6    9    5    0
         2     6       6    8    0    3
         3     9       6    7    5    0
         4    10       6    6    0    2
  5      1     2       7    9    6    0
         2     3       6    8    6    0
         3     4       6    7    6    0
         4     8       4    5    4    0
  6      1     2       9    8    6    0
         2     6       8    6    6    0
         3     6       9    4    0    9
         4     8       8    4    0    9
  7      1     1       8    6    5    0
         2     2       8    5    0    2
         3     6       6    5    0    2
         4    10       5    4    4    0
  8      1     2       6    9    6    0
         2     3       4    8    4    0
         3     5       3    7    3    0
         4     5       3    8    0    6
  9      1     1      10    8    3    0
         2     6       8    8    2    0
         3     9       4    8    0    8
         4    10       1    7    0    7
 10      1     4      10    8    0    9
         2     4      10    7    7    0
         3     6      10    7    6    0
         4    10       9    6    0    8
 11      1     6       7    9    6    0
         2     6       6    8    0    8
         3     8       5    6    0    5
         4     9       4    5    0    3
 12      1     2       9    6    5    0
         2     7       8    5    0    4
         3     8       6    4    0    2
         4    10       4    3    3    0
 13      1     1       6    8    0    8
         2     2       6    8    7    0
         3     4       6    8    0    7
         4     7       6    7    2    0
 14      1     1       7    6    4    0
         2     9       6    5    3    0
         3    10       4    5    0    5
         4    10       3    5    1    0
 15      1     1       9    9    4    0
         2     6       9    7    4    0
         3    10       8    3    0    4
         4    10       8    5    4    0
 16      1     4       6    4    4    0
         2     5       5    4    4    0
         3     8       5    3    2    0
         4    10       3    3    0    8
 17      1     4       8    8    0    2
         2     5       6    6    3    0
         3     5       6    7    0    1
         4     7       2    4    0    1
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   25   25   87   85
************************************************************************
