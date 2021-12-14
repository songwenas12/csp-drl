jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	15		2 3 4 5 6 7 9 10 11 12 13 14 16 20 33 
2	6	13		49 47 46 42 31 28 26 25 24 21 19 18 17 
3	6	10		49 47 45 43 29 28 27 24 23 15 
4	6	6		51 48 47 46 28 8 
5	6	15		50 48 47 46 45 44 43 40 38 37 36 32 31 30 28 
6	6	10		50 47 46 43 42 40 37 31 25 21 
7	6	12		49 48 47 45 43 42 38 37 36 31 30 28 
8	6	10		50 49 45 44 42 39 38 37 30 22 
9	6	9		43 42 40 38 37 36 31 30 28 
10	6	8		48 46 43 39 38 36 28 24 
11	6	9		42 41 40 38 36 35 31 30 26 
12	6	10		50 46 44 41 40 39 38 37 35 30 
13	6	9		44 43 41 40 38 37 36 31 30 
14	6	6		44 42 40 36 32 24 
15	6	8		44 42 38 37 36 34 32 31 
16	6	8		46 45 42 40 38 35 31 30 
17	6	7		45 40 38 37 36 35 30 
18	6	7		44 40 39 37 36 35 30 
19	6	6		43 40 37 36 32 30 
20	6	6		47 46 45 43 42 30 
21	6	5		41 38 36 32 30 
22	6	5		43 40 36 34 32 
23	6	5		38 37 36 34 31 
24	6	4		41 37 35 30 
25	6	3		48 41 34 
26	6	3		39 37 34 
27	6	3		42 40 34 
28	6	2		41 35 
29	6	2		38 36 
30	6	1		34 
31	6	1		39 
32	6	1		35 
33	6	1		38 
34	6	1		52 
35	6	1		52 
36	6	1		52 
37	6	1		52 
38	6	1		52 
39	6	1		52 
40	6	1		52 
41	6	1		52 
42	6	1		52 
43	6	1		52 
44	6	1		52 
45	6	1		52 
46	6	1		52 
47	6	1		52 
48	6	1		52 
49	6	1		52 
50	6	1		52 
51	6	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	1	5	4	14	16	17	20	
	2	2	5	4	13	14	16	18	
	3	5	5	3	10	14	15	16	
	4	9	5	3	10	13	15	16	
	5	10	5	2	6	13	15	15	
	6	14	5	2	6	12	14	14	
3	1	2	3	2	17	14	18	13	
	2	6	2	1	16	13	17	12	
	3	7	2	1	14	12	16	11	
	4	13	2	1	13	11	15	11	
	5	16	2	1	11	10	14	9	
	6	20	2	1	10	10	13	9	
4	1	1	3	4	18	16	3	20	
	2	5	2	4	18	14	3	17	
	3	9	2	4	15	14	3	17	
	4	10	2	3	13	13	3	15	
	5	12	2	3	13	12	3	14	
	6	20	2	3	11	12	3	13	
5	1	3	4	3	16	20	9	7	
	2	5	4	2	11	17	8	5	
	3	8	4	2	10	15	6	4	
	4	12	4	1	7	11	6	4	
	5	14	4	1	7	9	3	2	
	6	16	4	1	3	9	1	2	
6	1	1	4	5	18	19	8	12	
	2	2	4	4	18	15	7	10	
	3	7	3	3	18	14	7	8	
	4	8	3	2	17	11	7	4	
	5	9	2	2	16	11	7	3	
	6	15	1	1	16	9	7	2	
7	1	2	1	3	10	6	19	9	
	2	11	1	3	9	6	17	8	
	3	12	1	3	8	6	16	8	
	4	14	1	3	6	6	16	7	
	5	17	1	3	6	6	15	7	
	6	18	1	3	5	6	14	6	
8	1	1	2	4	8	20	7	3	
	2	3	2	4	7	16	7	2	
	3	8	2	3	6	16	7	2	
	4	9	1	2	5	12	6	2	
	5	12	1	1	4	11	5	2	
	6	16	1	1	4	9	5	2	
9	1	1	3	2	11	11	13	4	
	2	3	3	1	11	10	12	3	
	3	7	2	1	7	9	11	3	
	4	15	2	1	5	9	9	3	
	5	17	2	1	3	7	9	3	
	6	20	1	1	2	7	7	3	
10	1	13	4	1	18	9	15	18	
	2	15	3	1	17	9	14	15	
	3	16	3	1	17	8	11	15	
	4	18	3	1	17	7	7	14	
	5	19	3	1	17	6	7	11	
	6	20	3	1	17	6	3	10	
11	1	2	3	3	9	12	9	13	
	2	5	2	2	9	12	9	13	
	3	14	2	2	7	11	9	13	
	4	15	1	1	7	10	9	13	
	5	19	1	1	5	9	9	13	
	6	20	1	1	3	8	9	13	
12	1	1	4	4	3	15	7	17	
	2	2	4	3	3	12	6	17	
	3	11	4	3	3	12	5	15	
	4	15	4	3	2	9	4	14	
	5	18	4	2	2	8	2	14	
	6	19	4	2	1	5	1	13	
13	1	3	4	1	10	20	19	10	
	2	4	4	1	9	17	17	10	
	3	12	3	1	9	17	15	10	
	4	15	2	1	9	14	14	10	
	5	16	2	1	9	14	14	9	
	6	20	1	1	9	12	11	10	
14	1	1	5	4	15	16	6	15	
	2	12	4	4	13	15	5	12	
	3	13	4	4	11	15	4	11	
	4	14	3	4	11	14	4	9	
	5	19	2	4	9	14	4	7	
	6	20	2	4	7	14	3	5	
15	1	1	4	2	15	15	14	13	
	2	2	4	2	13	14	14	12	
	3	3	4	2	12	13	14	10	
	4	5	4	2	11	13	14	10	
	5	6	3	2	9	12	14	9	
	6	18	3	2	8	11	14	7	
16	1	1	1	4	15	18	17	20	
	2	10	1	4	15	16	16	16	
	3	13	1	4	13	15	16	15	
	4	14	1	4	10	13	16	11	
	5	16	1	4	9	11	15	7	
	6	17	1	4	6	11	15	7	
17	1	2	3	4	20	19	14	6	
	2	6	3	4	17	18	14	6	
	3	8	3	4	17	18	14	5	
	4	9	3	3	15	18	14	5	
	5	18	3	3	12	18	14	5	
	6	20	3	3	12	18	14	4	
18	1	4	4	4	16	13	7	17	
	2	8	3	4	14	11	7	12	
	3	10	3	4	11	10	7	9	
	4	14	2	4	7	9	7	7	
	5	17	2	4	4	8	7	5	
	6	18	1	4	3	8	7	2	
19	1	7	1	4	17	8	12	13	
	2	8	1	4	16	7	10	12	
	3	16	1	3	16	6	10	11	
	4	18	1	3	15	6	9	8	
	5	19	1	1	13	5	6	6	
	6	20	1	1	12	4	5	4	
20	1	2	1	3	19	19	7	17	
	2	5	1	3	16	18	7	17	
	3	14	1	3	14	16	6	14	
	4	16	1	3	13	15	4	8	
	5	19	1	3	11	15	3	6	
	6	20	1	3	10	14	3	4	
21	1	2	4	3	17	14	14	9	
	2	3	4	3	12	12	13	6	
	3	15	3	3	9	11	11	5	
	4	17	3	3	7	7	8	4	
	5	18	3	3	6	7	7	3	
	6	19	2	3	4	4	4	3	
22	1	1	3	3	19	12	15	11	
	2	3	2	2	17	12	12	9	
	3	5	2	2	16	9	11	9	
	4	9	1	2	15	6	8	6	
	5	12	1	1	14	4	8	6	
	6	15	1	1	11	1	4	4	
23	1	2	3	5	14	15	15	16	
	2	6	3	3	12	14	13	15	
	3	8	3	3	8	14	12	15	
	4	11	3	2	8	14	12	15	
	5	12	3	2	6	14	10	14	
	6	14	3	1	1	14	10	14	
24	1	1	4	2	13	11	12	16	
	2	2	3	1	10	9	11	11	
	3	4	3	1	10	8	11	9	
	4	7	3	1	9	5	11	6	
	5	12	1	1	7	4	11	4	
	6	16	1	1	6	2	11	4	
25	1	7	3	3	15	17	6	14	
	2	9	3	3	15	17	6	12	
	3	10	3	3	15	16	5	12	
	4	11	3	3	15	16	3	10	
	5	15	2	2	14	16	2	8	
	6	16	2	2	14	15	2	7	
26	1	5	3	4	19	6	9	20	
	2	10	2	4	14	6	6	13	
	3	11	2	3	12	6	5	10	
	4	14	1	3	11	6	3	7	
	5	17	1	1	7	5	2	6	
	6	20	1	1	5	5	1	4	
27	1	3	4	5	13	14	19	19	
	2	4	3	5	12	13	16	14	
	3	5	3	5	11	13	15	14	
	4	11	3	5	11	10	14	10	
	5	19	2	5	11	8	14	9	
	6	20	2	5	10	7	12	6	
28	1	1	2	5	20	20	14	17	
	2	2	1	4	19	17	14	15	
	3	3	1	4	17	17	14	14	
	4	8	1	4	17	16	13	10	
	5	11	1	4	15	14	13	8	
	6	17	1	4	15	13	12	6	
29	1	3	3	2	20	19	16	20	
	2	5	2	2	18	18	12	19	
	3	6	2	2	17	17	11	19	
	4	14	1	2	16	17	10	18	
	5	15	1	2	16	16	7	18	
	6	17	1	2	15	15	3	18	
30	1	1	5	3	17	10	11	11	
	2	2	4	3	14	8	11	9	
	3	5	3	3	12	7	10	8	
	4	9	3	3	9	5	10	7	
	5	11	2	3	7	2	9	3	
	6	12	2	3	5	1	9	1	
31	1	1	1	5	19	14	19	14	
	2	6	1	4	17	13	18	11	
	3	9	1	4	16	13	17	10	
	4	13	1	4	16	13	16	8	
	5	16	1	4	14	12	16	6	
	6	19	1	4	13	12	15	3	
32	1	5	5	4	14	12	4	14	
	2	7	4	4	14	11	4	12	
	3	8	3	4	13	9	4	8	
	4	10	3	3	12	8	3	8	
	5	16	2	2	12	7	3	4	
	6	17	1	2	11	7	3	4	
33	1	3	5	3	10	12	13	3	
	2	10	3	3	10	8	12	2	
	3	11	3	3	10	8	9	2	
	4	13	3	3	9	5	8	2	
	5	14	2	3	9	5	5	2	
	6	16	1	3	9	3	4	2	
34	1	1	4	4	11	10	16	6	
	2	3	3	3	8	9	14	6	
	3	4	3	3	7	9	10	6	
	4	9	2	2	5	9	6	7	
	5	10	2	2	5	9	6	6	
	6	15	1	1	2	9	3	6	
35	1	3	4	4	14	9	9	14	
	2	8	4	4	13	9	8	11	
	3	10	4	4	13	8	8	8	
	4	11	4	4	13	7	7	7	
	5	14	4	3	12	7	5	6	
	6	18	4	3	12	6	4	4	
36	1	7	2	5	13	10	19	16	
	2	10	2	4	12	9	18	16	
	3	11	2	4	10	9	16	16	
	4	14	2	4	7	8	15	16	
	5	18	2	3	6	7	15	16	
	6	19	2	3	4	7	14	16	
37	1	12	5	4	17	6	17	9	
	2	14	4	4	14	6	16	7	
	3	15	4	4	12	6	15	5	
	4	16	4	4	9	6	12	4	
	5	17	4	4	7	6	9	2	
	6	18	4	4	3	6	7	2	
38	1	3	5	3	13	6	17	11	
	2	4	4	3	13	5	15	11	
	3	10	4	3	12	4	13	11	
	4	15	3	2	12	3	9	11	
	5	19	2	2	11	2	9	11	
	6	20	2	2	11	1	6	11	
39	1	1	5	4	13	3	17	15	
	2	2	4	4	11	3	17	13	
	3	5	4	4	9	3	17	11	
	4	14	3	4	8	3	16	10	
	5	16	3	4	6	3	16	9	
	6	17	3	4	4	3	16	7	
40	1	1	2	4	6	19	9	2	
	2	7	2	4	6	17	9	2	
	3	8	2	3	4	17	9	2	
	4	10	2	3	4	14	9	2	
	5	15	1	2	3	14	9	2	
	6	17	1	1	2	13	9	2	
41	1	3	3	4	14	15	14	8	
	2	4	3	4	12	12	12	7	
	3	5	3	4	8	10	11	6	
	4	8	3	4	8	10	8	4	
	5	15	2	4	4	8	6	3	
	6	16	2	4	2	6	5	3	
42	1	4	2	5	18	16	8	16	
	2	5	2	4	15	12	7	15	
	3	8	2	4	14	10	7	14	
	4	9	2	4	11	9	6	10	
	5	10	2	4	6	7	5	9	
	6	11	2	4	6	4	5	6	
43	1	1	3	3	15	4	16	15	
	2	2	3	3	12	3	15	14	
	3	10	3	2	10	3	15	13	
	4	11	3	2	8	3	15	13	
	5	16	3	2	6	3	15	13	
	6	17	3	1	6	3	15	12	
44	1	1	4	5	16	12	17	10	
	2	4	4	4	15	11	16	8	
	3	5	4	4	15	7	12	6	
	4	16	4	4	14	7	10	6	
	5	19	3	4	14	3	9	4	
	6	20	3	4	14	1	6	4	
45	1	2	3	3	7	12	3	9	
	2	4	3	2	7	10	2	7	
	3	6	3	2	6	10	2	6	
	4	7	3	2	6	9	2	6	
	5	15	3	2	4	8	1	5	
	6	16	3	2	4	8	1	3	
46	1	1	5	4	16	1	9	19	
	2	2	4	3	11	1	9	18	
	3	6	4	3	11	1	9	16	
	4	9	3	2	6	1	9	14	
	5	14	2	1	4	1	9	12	
	6	17	2	1	4	1	9	11	
47	1	1	2	4	17	17	7	13	
	2	6	2	4	15	12	6	12	
	3	7	2	4	14	11	6	12	
	4	8	2	4	13	8	5	12	
	5	11	2	3	12	4	5	11	
	6	16	2	3	11	3	4	11	
48	1	5	3	2	16	18	12	13	
	2	6	3	2	16	17	10	12	
	3	9	3	2	16	15	9	12	
	4	14	2	2	15	12	9	12	
	5	16	2	2	15	9	8	12	
	6	19	2	2	14	7	7	12	
49	1	1	4	4	8	14	17	8	
	2	2	3	3	7	12	15	8	
	3	4	3	3	6	10	13	8	
	4	5	3	2	5	6	11	8	
	5	11	3	2	4	4	8	8	
	6	16	3	1	4	3	7	8	
50	1	2	1	4	18	15	10	16	
	2	9	1	3	17	14	7	13	
	3	10	1	3	15	10	5	13	
	4	15	1	3	13	9	5	11	
	5	19	1	3	11	6	4	6	
	6	20	1	3	9	3	1	5	
51	1	4	5	5	4	14	18	11	
	2	12	4	4	4	12	16	11	
	3	13	3	4	3	8	14	11	
	4	15	2	4	3	8	9	11	
	5	18	2	4	2	5	8	10	
	6	19	1	4	1	2	6	10	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	33	30	546	516	493	491

************************************************************************
