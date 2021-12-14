************************************************************************
file with basedata            : mf44_.bas
initial value random generator: 20651
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  32
horizon                       :  239
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     30      0       41        0       41
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           6  14
   3        3          3           7  13  18
   4        3          3           5   6  18
   5        3          3           8   9  10
   6        3          1          11
   7        3          2          17  21
   8        3          1          30
   9        3          2          12  26
  10        3          2          14  26
  11        3          3          12  20  22
  12        3          2          13  15
  13        3          1          21
  14        3          1          15
  15        3          2          16  17
  16        3          2          21  24
  17        3          3          23  24  28
  18        3          3          19  24  25
  19        3          1          22
  20        3          2          27  28
  21        3          2          23  29
  22        3          1          30
  23        3          1          30
  24        3          2          27  31
  25        3          3          26  27  28
  26        3          2          29  31
  27        3          1          29
  28        3          1          31
  29        3          1          32
  30        3          1          32
  31        3          1          32
  32        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       8    0    6   10
         2     3       0    6    4    9
         3     4       0    4    3    9
  3      1     3       0   10    9    8
         2     5       0    5    7    4
         3    10       3    0    2    4
  4      1     9       0    9    7    8
         2    10       8    0    5    4
         3    10       8    0    3    5
  5      1     4       0    3    4    2
         2     6       0    2    3    1
         3     6       9    0    3    1
  6      1     2       7    0    8   10
         2     5       6    0    6    7
         3     7       0    8    6    7
  7      1     1       0    5    4    5
         2     3       0    4    4    2
         3     9       3    0    3    1
  8      1     2       9    0   10    6
         2     2       0   10   10    6
         3    10       8    0    6    4
  9      1     4       7    0    4    8
         2     7       3    0    2    8
         3     7       0    6    2    7
 10      1     7       0    5    6    1
         2     7       0    2    6    3
         3     7       5    0    7    1
 11      1     1       0    1    5    9
         2     6      10    0    5    5
         3     6       0    1    5    4
 12      1     1       0    5    8    7
         2     4       8    0    5    6
         3     8       0    3    4    5
 13      1     2       8    0    6    6
         2     3       6    0    5    5
         3     7       0    7    5    5
 14      1     1       0    4    8    6
         2     5       4    0    8    6
         3     6       4    0    6    4
 15      1     8       0    6    6    9
         2    10       7    0    4    8
         3    10       3    0    2    9
 16      1     2       6    0    6    5
         2     2       0    5    6    4
         3     4       5    0    5    4
 17      1     3       0    4    2    4
         2     8       7    0    2    3
         3    10       6    0    2    2
 18      1     1       7    0    7    6
         2     3       7    0    6    6
         3    10       2    0    3    6
 19      1     2       5    0    7    4
         2     3       3    0    7    2
         3     3       2    0    7    4
 20      1     3       9    0    9    4
         2     5       8    0    6    4
         3    10       5    0    2    4
 21      1     2       0    9    8    7
         2     3       0    7    7    6
         3     8       0    6    5    4
 22      1     5       0    7    6    4
         2     9       0    5    6    3
         3    10       0    2    4    1
 23      1     7       8    0    9    9
         2     9       8    0    5    6
         3     9       8    0    6    4
 24      1     1       5    0   10    9
         2     4       2    0    9    8
         3     9       2    0    8    5
 25      1     4       7    0    8    8
         2     9       0    7    6    7
         3    10       0    3    6    7
 26      1     3       0   10    7    5
         2     5       0    9    5    4
         3     7       0    9    2    2
 27      1     2       3    0    3    5
         2     6       0    6    2    2
         3    10       3    0    1    2
 28      1     7       0    9    6    3
         2     8       0    7    5    3
         3    10       0    5    2    1
 29      1     4       6    0    9    4
         2     6       0    7    9    3
         3    10       0    5    9    1
 30      1     1      10    0    2    4
         2     5       0    5    2    4
         3     6       5    0    1    4
 31      1     1       1    0    6    4
         2     5       0    3    5    4
         3     6       1    0    5    2
 32      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   34   36  160  149
************************************************************************
