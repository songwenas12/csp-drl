jobs  (incl. supersource/sink ):	102
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 6 7 9 10 11 13 
2	3	8		30 29 25 23 22 19 16 14 
3	3	5		23 22 19 15 8 
4	3	7		30 29 25 22 20 16 15 
5	3	7		30 27 26 25 20 18 16 
6	3	4		32 29 27 12 
7	3	5		30 21 20 19 15 
8	3	6		30 29 28 27 25 16 
9	3	5		30 27 22 17 16 
10	3	8		32 30 28 27 26 24 22 21 
11	3	6		29 27 26 24 20 17 
12	3	6		35 30 26 25 24 17 
13	3	4		36 30 27 16 
14	3	6		31 28 27 26 21 18 
15	3	9		40 38 36 35 33 31 27 26 24 
16	3	5		35 32 31 24 21 
17	3	4		36 31 28 21 
18	3	10		43 42 41 40 39 38 36 35 34 33 
19	3	9		43 41 40 39 36 35 33 32 27 
20	3	6		39 36 34 33 31 28 
21	3	7		43 42 40 39 38 34 33 
22	3	5		42 41 36 35 31 
23	3	5		46 42 41 39 31 
24	3	5		42 41 39 37 34 
25	3	4		46 41 36 31 
26	3	7		51 50 49 43 42 39 37 
27	3	8		55 52 51 50 49 46 42 37 
28	3	6		51 50 46 42 41 37 
29	3	5		49 43 42 35 34 
30	3	8		52 50 48 47 46 45 42 38 
31	3	7		55 52 51 50 49 44 37 
32	3	6		55 52 50 46 42 37 
33	3	6		55 51 50 49 44 37 
34	3	8		57 56 55 52 50 46 45 44 
35	3	11		62 59 57 56 55 54 53 52 50 47 46 
36	3	11		63 62 59 57 56 55 54 53 51 50 47 
37	3	7		63 58 57 56 48 47 45 
38	3	7		62 57 56 55 54 49 44 
39	3	4		55 52 45 44 
40	3	11		63 62 61 60 59 58 57 56 55 54 53 
41	3	7		63 57 56 55 53 49 47 
42	3	3		64 56 44 
43	3	7		64 62 61 60 56 54 53 
44	3	7		65 63 61 60 59 58 53 
45	3	6		65 62 61 60 54 53 
46	3	8		73 70 66 65 63 61 60 58 
47	3	9		76 73 72 71 70 66 65 64 60 
48	3	7		72 71 70 67 65 64 59 
49	3	7		73 72 66 65 64 61 60 
50	3	7		76 72 70 67 66 61 58 
51	3	7		73 72 71 68 67 66 65 
52	3	5		87 73 65 63 60 
53	3	7		73 72 71 70 69 68 67 
54	3	9		81 80 79 76 75 73 71 67 66 
55	3	6		87 72 69 67 65 64 
56	3	6		76 70 69 68 67 65 
57	3	6		87 76 74 73 71 65 
58	3	6		89 80 75 74 69 68 
59	3	6		82 80 78 76 73 66 
60	3	5		81 80 75 69 67 
61	3	7		89 87 85 84 81 74 71 
62	3	7		86 84 81 79 78 76 70 
63	3	3		75 72 68 
64	3	4		89 84 75 68 
65	3	8		86 84 82 81 79 78 77 75 
66	3	5		89 87 85 74 69 
67	3	6		85 84 83 82 78 74 
68	3	7		90 86 83 82 81 79 78 
69	3	6		101 92 88 86 84 77 
70	3	5		89 88 87 83 75 
71	3	5		92 90 82 78 77 
72	3	4		92 86 80 77 
73	3	5		101 91 89 85 84 
74	3	6		101 100 92 91 88 86 
75	3	6		101 99 93 92 91 85 
76	3	5		99 92 91 90 89 
77	3	3		100 91 83 
78	3	5		101 99 97 94 91 
79	3	5		101 100 98 95 88 
80	3	3		99 93 85 
81	3	5		100 98 97 95 94 
82	3	4		98 97 95 94 
83	3	3		97 94 93 
84	3	2		99 90 
85	3	3		97 95 94 
86	3	3		98 97 93 
87	3	3		97 96 93 
88	3	3		99 97 96 
89	3	3		100 97 95 
90	3	2		96 93 
91	3	2		98 95 
92	3	2		96 95 
93	3	1		95 
94	3	1		96 
95	3	1		102 
96	3	1		102 
97	3	1		102 
98	3	1		102 
99	3	1		102 
100	3	1		102 
101	3	1		102 
102	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	1	0	3	8	6	
	2	6	2	0	6	5	
	3	7	2	0	2	4	
3	1	3	0	4	3	4	
	2	9	4	0	3	2	
	3	10	4	0	3	1	
4	1	4	1	0	8	9	
	2	6	0	3	6	7	
	3	9	1	0	4	5	
5	1	7	0	5	4	10	
	2	7	3	0	2	8	
	3	8	0	2	1	8	
6	1	4	5	0	8	3	
	2	7	5	0	4	3	
	3	10	5	0	2	2	
7	1	1	0	2	7	9	
	2	3	3	0	4	9	
	3	9	3	0	2	7	
8	1	4	4	0	8	3	
	2	5	3	0	4	3	
	3	10	0	1	2	3	
9	1	5	0	5	6	10	
	2	7	0	3	6	5	
	3	10	2	0	6	2	
10	1	2	0	5	7	9	
	2	2	2	0	6	6	
	3	3	0	3	6	5	
11	1	2	2	0	5	4	
	2	3	0	1	3	4	
	3	4	0	1	2	3	
12	1	2	0	5	9	7	
	2	6	0	4	6	6	
	3	10	3	0	5	6	
13	1	2	3	0	7	5	
	2	3	3	0	4	3	
	3	5	0	1	4	2	
14	1	2	4	0	7	4	
	2	3	0	5	6	3	
	3	7	4	0	5	1	
15	1	1	4	0	5	9	
	2	5	4	0	4	4	
	3	8	4	0	4	1	
16	1	3	4	0	8	3	
	2	4	4	0	7	1	
	3	9	0	3	7	1	
17	1	7	0	5	6	3	
	2	7	2	0	5	2	
	3	9	1	0	5	2	
18	1	2	0	2	5	3	
	2	6	0	2	3	2	
	3	10	0	2	2	2	
19	1	6	3	0	8	3	
	2	7	2	0	6	3	
	3	9	2	0	6	2	
20	1	4	1	0	7	1	
	2	7	0	3	4	1	
	3	10	1	0	3	1	
21	1	2	0	2	7	9	
	2	8	3	0	7	7	
	3	9	0	1	7	4	
22	1	3	0	2	7	10	
	2	7	0	2	5	10	
	3	10	3	0	5	10	
23	1	4	0	1	7	6	
	2	5	0	1	5	6	
	3	9	1	0	4	6	
24	1	2	1	0	3	9	
	2	5	0	2	2	4	
	3	8	1	0	2	1	
25	1	6	0	5	7	8	
	2	8	0	3	6	7	
	3	8	2	0	5	6	
26	1	1	0	4	9	7	
	2	5	0	3	6	6	
	3	7	0	3	5	6	
27	1	2	5	0	6	8	
	2	9	4	0	5	7	
	3	10	4	0	3	6	
28	1	2	0	2	6	4	
	2	6	0	1	3	4	
	3	7	0	1	3	3	
29	1	1	0	1	6	8	
	2	2	0	1	3	5	
	3	9	1	0	1	2	
30	1	5	3	0	7	7	
	2	8	3	0	5	5	
	3	9	0	2	1	1	
31	1	6	0	5	7	9	
	2	7	4	0	6	8	
	3	8	0	2	6	8	
32	1	1	4	0	6	5	
	2	7	0	2	6	5	
	3	7	2	0	6	5	
33	1	1	2	0	7	5	
	2	4	1	0	6	3	
	3	5	1	0	5	2	
34	1	4	0	4	6	10	
	2	6	0	4	3	8	
	3	10	0	4	2	8	
35	1	5	3	0	9	8	
	2	7	0	3	7	5	
	3	8	0	1	7	2	
36	1	5	0	1	10	6	
	2	6	3	0	7	6	
	3	9	0	1	6	6	
37	1	3	0	4	7	7	
	2	3	2	0	6	5	
	3	9	0	3	6	4	
38	1	1	0	1	3	4	
	2	4	0	1	1	3	
	3	8	0	1	1	2	
39	1	2	0	2	10	3	
	2	6	0	2	7	2	
	3	8	0	2	5	2	
40	1	1	3	0	7	9	
	2	4	0	2	7	5	
	3	8	1	0	3	3	
41	1	5	0	1	4	7	
	2	6	0	1	3	5	
	3	9	0	1	3	4	
42	1	9	2	0	5	9	
	2	9	0	2	4	8	
	3	10	0	1	3	6	
43	1	5	0	3	7	7	
	2	7	0	3	6	5	
	3	10	4	0	6	5	
44	1	2	0	3	4	5	
	2	3	0	2	4	3	
	3	9	2	0	3	3	
45	1	2	0	3	5	9	
	2	2	2	0	5	9	
	3	4	0	3	3	9	
46	1	4	4	0	6	1	
	2	7	0	4	4	1	
	3	10	0	3	1	1	
47	1	2	3	0	7	10	
	2	7	0	3	7	9	
	3	10	2	0	4	7	
48	1	5	0	2	9	5	
	2	6	0	2	6	4	
	3	8	3	0	5	3	
49	1	5	2	0	6	2	
	2	6	2	0	5	2	
	3	10	2	0	5	1	
50	1	1	0	5	6	10	
	2	8	0	3	6	6	
	3	9	4	0	6	4	
51	1	1	4	0	3	5	
	2	4	0	2	3	4	
	3	4	3	0	3	1	
52	1	3	1	0	2	7	
	2	8	1	0	2	5	
	3	9	1	0	2	2	
53	1	6	2	0	9	5	
	2	8	0	3	8	5	
	3	10	2	0	8	5	
54	1	1	0	3	8	5	
	2	2	0	3	8	4	
	3	3	0	2	8	3	
55	1	5	4	0	9	8	
	2	7	0	4	8	8	
	3	8	1	0	7	6	
56	1	2	0	5	9	9	
	2	6	1	0	9	5	
	3	7	1	0	8	3	
57	1	5	0	3	7	3	
	2	6	0	3	4	3	
	3	10	1	0	4	1	
58	1	6	4	0	7	8	
	2	7	4	0	6	6	
	3	9	4	0	4	3	
59	1	3	4	0	6	7	
	2	5	3	0	4	5	
	3	7	2	0	4	4	
60	1	1	3	0	7	2	
	2	6	0	3	6	1	
	3	9	3	0	4	1	
61	1	3	0	4	8	7	
	2	4	0	2	5	5	
	3	6	1	0	4	1	
62	1	2	4	0	5	5	
	2	3	3	0	4	5	
	3	10	0	2	3	5	
63	1	3	4	0	6	5	
	2	5	0	1	4	3	
	3	9	0	1	3	3	
64	1	3	2	0	8	8	
	2	7	1	0	7	7	
	3	7	0	2	7	3	
65	1	8	0	5	9	3	
	2	10	0	4	7	1	
	3	10	2	0	7	1	
66	1	4	2	0	4	10	
	2	5	0	3	3	8	
	3	6	2	0	3	8	
67	1	5	1	0	6	6	
	2	5	0	1	6	6	
	3	6	0	1	6	5	
68	1	5	0	3	3	9	
	2	6	3	0	2	5	
	3	8	3	0	2	4	
69	1	8	2	0	2	2	
	2	9	2	0	1	2	
	3	10	2	0	1	1	
70	1	4	4	0	6	9	
	2	6	0	1	5	8	
	3	7	3	0	1	8	
71	1	1	0	5	8	2	
	2	2	2	0	7	1	
	3	7	2	0	4	1	
72	1	4	0	2	8	8	
	2	4	4	0	4	8	
	3	5	3	0	3	8	
73	1	4	0	4	7	4	
	2	5	3	0	6	4	
	3	8	0	4	6	4	
74	1	1	0	1	6	8	
	2	6	4	0	4	6	
	3	7	4	0	3	3	
75	1	2	0	5	9	7	
	2	3	3	0	7	4	
	3	5	2	0	5	4	
76	1	5	3	0	9	8	
	2	7	2	0	6	7	
	3	9	0	1	2	6	
77	1	3	3	0	7	9	
	2	7	3	0	5	9	
	3	8	2	0	5	8	
78	1	1	3	0	8	8	
	2	6	3	0	5	5	
	3	9	0	1	4	4	
79	1	5	0	3	7	9	
	2	6	0	2	5	9	
	3	8	2	0	4	9	
80	1	6	4	0	10	1	
	2	8	0	2	9	1	
	3	9	0	1	9	1	
81	1	2	0	3	6	2	
	2	5	3	0	6	2	
	3	9	2	0	6	2	
82	1	5	4	0	7	5	
	2	8	3	0	7	2	
	3	9	1	0	6	2	
83	1	3	3	0	6	3	
	2	5	0	2	4	3	
	3	9	0	2	4	2	
84	1	6	0	5	8	7	
	2	6	5	0	7	7	
	3	8	5	0	5	6	
85	1	2	0	3	8	8	
	2	7	0	3	6	8	
	3	10	0	3	3	6	
86	1	3	0	5	3	6	
	2	3	5	0	2	4	
	3	7	5	0	2	3	
87	1	1	4	0	6	2	
	2	5	3	0	6	2	
	3	6	0	1	4	2	
88	1	1	0	5	7	2	
	2	4	2	0	3	3	
	3	5	2	0	3	2	
89	1	2	0	2	9	4	
	2	5	0	2	8	2	
	3	10	4	0	8	2	
90	1	2	5	0	8	5	
	2	8	5	0	6	4	
	3	10	5	0	4	3	
91	1	1	0	5	5	5	
	2	2	0	3	4	4	
	3	5	0	2	4	3	
92	1	4	0	4	7	9	
	2	4	2	0	3	7	
	3	9	1	0	3	6	
93	1	5	5	0	7	9	
	2	8	5	0	7	8	
	3	9	5	0	5	8	
94	1	2	4	0	6	5	
	2	3	0	1	5	4	
	3	6	3	0	5	3	
95	1	3	5	0	7	8	
	2	6	0	1	4	7	
	3	9	1	0	3	7	
96	1	3	4	0	8	5	
	2	7	0	2	7	3	
	3	10	0	2	5	2	
97	1	1	0	1	4	6	
	2	6	2	0	3	5	
	3	8	0	1	1	5	
98	1	1	5	0	8	8	
	2	2	0	3	6	6	
	3	10	1	0	3	4	
99	1	3	0	2	5	5	
	2	5	3	0	5	4	
	3	6	2	0	5	4	
100	1	4	0	4	3	9	
	2	5	0	4	3	5	
	3	6	0	4	2	5	
101	1	1	0	1	3	7	
	2	3	0	1	2	7	
	3	7	5	0	1	7	
102	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	11	11	470	448

************************************************************************
