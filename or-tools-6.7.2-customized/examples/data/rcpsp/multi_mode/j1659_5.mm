************************************************************************
file with basedata            : md251_.bas
initial value random generator: 1500555738
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  18
horizon                       :  134
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     16      0       19       14       19
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6   9
   3        3          3           7  10  12
   4        3          3           5   6  16
   5        3          2           8  13
   6        3          2          14  15
   7        3          2           8  13
   8        3          1          11
   9        3          3          10  11  15
  10        3          2          16  17
  11        3          1          14
  12        3          2          13  16
  13        3          2          14  15
  14        3          1          17
  15        3          1          18
  16        3          1          18
  17        3          1          18
  18        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       5    0    6    5
         2     3       0    8    7    5
         3     6       0    6    5    3
  3      1     2       0    4    2    8
         2     6       0    3    2    6
         3     9       0    2    2    6
  4      1     4       5    0    4    4
         2     5       0    3    2    4
         3     6       5    0    1    3
  5      1     1       7    0    8    5
         2     3       0    9    7    5
         3     6       0    4    6    5
  6      1     1       0    7   10    5
         2     8       0    6   10    4
         3     9       0    5   10    2
  7      1     1       0    3    3   10
         2    10       2    0    3    7
         3    10       0    2    2    8
  8      1     4       0    3    7   10
         2     6       0    3    5    8
         3    10       0    2    5    7
  9      1     2       0    6    8    7
         2     5       7    0    7    7
         3    10       4    0    5    7
 10      1     3       6    0    6   10
         2     7       6    0    4    9
         3     9       6    0    2    7
 11      1     8       0    3    9    3
         2     8       4    0    7    3
         3    10       4    0    6    3
 12      1     3       5    0    3    8
         2     5       0    9    3    7
         3     9       2    0    3    4
 13      1     3       5    0    7    7
         2     7       0    8    6    6
         3     9       0    7    4    3
 14      1     1       1    0    9    9
         2     6       0    4    6    8
         3     8       1    0    5    7
 15      1     3       0    8    8   10
         2     7       7    0    3    8
         3     7       0    8    4    8
 16      1     1       0    6    8    4
         2     5       0    6    8    3
         3     6       0    5    8    3
 17      1     1       7    0    8    6
         2     6       7    0    7    5
         3    10       0    3    4    1
 18      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   20  107  111
************************************************************************
