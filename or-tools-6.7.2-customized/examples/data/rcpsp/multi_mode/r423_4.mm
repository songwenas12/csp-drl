************************************************************************
file with basedata            : cr423_.bas
initial value random generator: 1891217455
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  129
RESOURCES
  - renewable                 :  4   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       21        6       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   8  13
   3        3          2          10  14
   4        3          3           7  12  13
   5        3          3           6   7   9
   6        3          1          12
   7        3          2          14  16
   8        3          2          11  15
   9        3          3          12  14  16
  10        3          3          11  13  15
  11        3          1          16
  12        3          2          15  17
  13        3          1          17
  14        3          1          17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  R 4  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0    0    0
  2      1     2       9   10    2    8    2    0
         2     9       6    9    2    5    0    9
         3     9       6   10    2    4    0    7
  3      1     1       8    7    9    9    0    9
         2     3       6    4    7    8    0    8
         3     9       5    2    7    8    0    5
  4      1     1       5    9    4    7    0    7
         2     2       3    5    3    6    6    0
         3     6       2    4    1    4    0    5
  5      1     2       3    4    6    9    0    9
         2     3       2    3    4    8    6    0
         3     4       1    1    3    3    4    0
  6      1     2       8    9    3    5    0    8
         2     6       8    6    2    5    7    0
         3     8       7    6    1    4    7    0
  7      1     5       7    7    8   10    0    8
         2     7       5    5    7    9    0    6
         3    10       3    5    7    9    8    0
  8      1     8       8    7    4    7    6    0
         2     9       6    6    4    7    3    0
         3     9       5    6    4    6    0    7
  9      1     2       2    6    3    4    4    0
         2     3       2    5    3    4    0    9
         3     8       2    4    2    3    4    0
 10      1     1       6    6    8    7    9    0
         2     6       5    5    5    5    0    6
         3     8       4    4    5    5    0    2
 11      1     6       8    4    5    8    0    7
         2     9       8    3    4    7    9    0
         3    10       7    2    4    7    0    5
 12      1     2       6    8    5    8    3    0
         2     3       6    8    4    8    0    6
         3    10       6    8    4    7    3    0
 13      1     4       6    3    5    9    6    0
         2     8       5    2    5    8    4    0
         3     9       3    2    4    7    3    0
 14      1     8      10    9    3    8    7    0
         2     9       9    8    2    5    0    6
         3    10       9    7    1    4    4    0
 15      1     5       6    5    2    7    0    6
         2    10       6    5    1    7    0    6
         3    10       5    5    2    7    0    4
 16      1     1       6    3    2    3    0    5
         2     1       3    2    2    3    0    6
         3     1       4    2    1    3    7    0
 17      1     4       4    5    9    7    5    0
         2     6       4    4    8    7    5    0
         3     8       3    1    8    5    0    8
 18      1     0       0    0    0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  R 4  N 1  N 2
   26   26   19   29   65   86
************************************************************************
