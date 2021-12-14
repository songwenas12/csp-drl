************************************************************************
file with basedata            : cr432_.bas
initial value random generator: 2031058238
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  126
RESOURCES
  - renewable                 :  4   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       23        5       23
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   7   9
   3        3          2           6  11
   4        3          2           6   7
   5        3          3          11  12  13
   6        3          2          15  16
   7        3          3           8  10  11
   8        3          2          12  15
   9        3          2          12  17
  10        3          2          14  17
  11        3          1          14
  12        3          1          14
  13        3          3          15  16  17
  14        3          1          16
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  R 4  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0    0    0
  2      1     2       5    6    8   10    7    0
         2     4       4    6    7    9    7    0
         3     7       4    5    6    7    0    5
  3      1     5       8   10    9    5    6    0
         2     8       8    5    8    3    6    0
         3     9       7    5    8    2    0    9
  4      1     4       9    2    7    8    5    0
         2     5       9    1    5    8    0    5
         3     6       9    1    5    5    0    5
  5      1     1       2    7    7    5    7    0
         2     2       1    4    5    4    6    0
         3     7       1    1    4    4    6    0
  6      1     6       6    9    9    4    0    8
         2     6       6    8    9    4    5    0
         3     7       5    8    8    3    2    0
  7      1     3      10    7    3    3    6    0
         2     4       6    6    3    2    0    5
         3     8       3    5    2    1    5    0
  8      1     5      10    4    8    7    8    0
         2     6       9    3    6    5    5    0
         3     9       9    3    5    5    2    0
  9      1     9       6    6   10    4    5    0
         2    10       2    5    9    2    0   10
         3    10       5    4   10    1    2    0
 10      1     1       3    2    4    8    0    4
         2     8       2    2    4    4    9    0
         3     9       1    2    3    3    6    0
 11      1     4       9    9    9    5    5    0
         2     5       8    9    7    4    4    0
         3     8       7    7    7    2    0    3
 12      1     6       5    8    8    9    6    0
         2     7       2    7    7    7    3    0
         3     8       2    6    6    7    1    0
 13      1     4       7    7   10    7    0    5
         2     7       7    7    8    6   10    0
         3     9       5    6    8    6    0    4
 14      1     1       5    9    9    8    0    5
         2     7       3    8    4    7    0    3
         3     7       2    9    5    5    1    0
 15      1     4      10    6    6    3    0    3
         2     7       8    6    6    3    9    0
         3     8       3    6    6    3    8    0
 16      1     4       6    9    5   10    0    7
         2     5       6    9    4    9    0    5
         3     6       5    8    4    7    0    4
 17      1     6       8    7    9    9    7    0
         2     7       8    7    8    8    0    8
         3     8       6    7    8    6    5    0
 18      1     0       0    0    0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  R 4  N 1  N 2
   34   30   40   28   96   77
************************************************************************
