jobs  (incl. supersource/sink ):	102
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	9		2 3 4 5 6 7 8 10 17 
2	3	5		25 22 15 12 9 
3	3	6		21 16 15 14 13 11 
4	3	6		31 24 23 18 15 12 
5	3	12		35 31 29 28 27 26 25 24 21 20 19 18 
6	3	8		30 29 25 23 22 19 18 13 
7	3	8		35 31 28 26 23 20 16 14 
8	3	9		35 31 28 26 25 23 21 19 14 
9	3	8		35 31 28 21 20 19 18 14 
10	3	8		35 30 29 28 27 24 19 18 
11	3	6		35 28 27 20 19 18 
12	3	8		38 35 32 29 28 27 21 20 
13	3	9		38 36 34 33 31 28 27 26 24 
14	3	8		43 38 36 34 33 29 27 24 
15	3	4		36 27 26 19 
16	3	3		36 29 19 
17	3	6		49 36 32 31 27 23 
18	3	10		49 43 41 40 38 37 36 34 33 32 
19	3	9		49 43 41 40 38 37 34 33 32 
20	3	7		49 47 40 39 36 33 30 
21	3	7		49 43 40 39 37 36 33 
22	3	4		43 42 36 28 
23	3	8		48 46 43 42 41 40 39 38 
24	3	7		49 44 42 41 40 39 32 
25	3	7		49 47 44 43 41 40 32 
26	3	7		49 48 46 44 43 40 39 
27	3	5		48 47 44 41 37 
28	3	6		49 48 46 44 40 39 
29	3	6		50 49 48 47 44 39 
30	3	4		67 48 46 41 
31	3	3		48 46 40 
32	3	4		67 48 46 45 
33	3	4		67 50 48 42 
34	3	3		50 44 39 
35	3	3		47 46 45 
36	3	3		50 45 44 
37	3	3		59 46 42 
38	3	5		55 53 52 50 44 
39	3	5		67 59 54 51 45 
40	3	4		67 54 51 45 
41	3	4		59 51 50 45 
42	3	3		54 51 45 
43	3	3		54 51 45 
44	3	10		67 66 63 62 61 59 58 56 54 51 
45	3	9		65 62 61 60 58 56 55 53 52 
46	3	6		62 61 56 54 51 50 
47	3	7		67 62 61 59 56 55 51 
48	3	7		65 59 58 57 56 55 54 
49	3	7		65 64 62 59 57 56 55 
50	3	9		76 75 71 69 66 63 60 58 57 
51	3	8		76 75 69 68 65 64 60 57 
52	3	9		84 82 79 75 74 71 69 66 64 
53	3	5		69 68 64 63 57 
54	3	9		82 79 77 76 75 74 72 71 70 
55	3	6		79 76 74 72 68 66 
56	3	7		84 79 77 75 72 71 70 
57	3	8		89 82 81 79 77 74 72 70 
58	3	4		84 78 74 64 
59	3	7		82 81 78 77 76 72 69 
60	3	8		89 84 81 80 78 74 73 70 
61	3	7		101 84 82 80 79 73 71 
62	3	9		101 100 88 87 84 82 81 78 77 
63	3	7		89 83 81 80 78 73 70 
64	3	6		101 89 88 81 77 72 
65	3	8		101 100 87 84 82 81 78 77 
66	3	6		89 83 81 78 73 70 
67	3	6		87 84 83 80 78 70 
68	3	7		101 100 87 86 84 82 77 
69	3	3		101 83 73 
70	3	10		101 100 98 97 96 91 90 88 86 85 
71	3	7		97 94 89 88 85 83 81 
72	3	2		80 73 
73	3	9		100 98 97 96 95 91 90 87 86 
74	3	9		101 99 98 97 96 94 88 85 83 
75	3	6		101 99 97 90 88 78 
76	3	6		100 99 91 87 85 84 
77	3	5		98 91 85 83 80 
78	3	4		98 92 91 85 
79	3	3		91 90 85 
80	3	5		97 96 94 93 92 
81	3	3		96 92 86 
82	3	3		97 94 92 
83	3	2		93 90 
84	3	2		96 90 
85	3	2		95 93 
86	3	2		99 93 
87	3	2		94 92 
88	3	2		95 92 
89	3	2		98 92 
90	3	1		92 
91	3	1		94 
92	3	1		102 
93	3	1		102 
94	3	1		102 
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
2	1	1	0	4	0	5	
	2	2	0	2	0	5	
	3	8	0	1	0	2	
3	1	9	0	5	0	8	
	2	10	2	0	5	0	
	3	10	0	3	0	6	
4	1	1	0	1	0	5	
	2	4	3	0	1	0	
	3	8	0	1	0	1	
5	1	4	0	2	3	0	
	2	7	4	0	2	0	
	3	9	3	0	2	0	
6	1	1	0	5	6	0	
	2	1	3	0	5	0	
	3	6	2	0	3	0	
7	1	2	0	3	0	8	
	2	7	0	2	3	0	
	3	8	2	0	2	0	
8	1	2	4	0	0	4	
	2	3	0	4	3	0	
	3	6	0	3	0	2	
9	1	6	4	0	9	0	
	2	7	3	0	8	0	
	3	8	2	0	7	0	
10	1	6	2	0	9	0	
	2	7	0	1	7	0	
	3	10	0	1	5	0	
11	1	1	0	5	7	0	
	2	2	3	0	0	5	
	3	2	3	0	4	0	
12	1	5	3	0	0	9	
	2	7	3	0	0	7	
	3	8	2	0	0	7	
13	1	8	5	0	0	6	
	2	8	5	0	4	0	
	3	9	5	0	0	2	
14	1	2	4	0	0	3	
	2	5	4	0	7	0	
	3	10	3	0	0	2	
15	1	1	5	0	0	9	
	2	5	4	0	0	6	
	3	9	4	0	0	5	
16	1	4	3	0	0	7	
	2	4	0	4	8	0	
	3	10	0	2	0	3	
17	1	2	0	4	0	6	
	2	4	0	2	0	5	
	3	6	1	0	0	5	
18	1	7	2	0	0	10	
	2	8	2	0	4	0	
	3	8	2	0	0	9	
19	1	1	0	2	8	0	
	2	6	1	0	8	0	
	3	10	0	1	8	0	
20	1	2	0	3	0	8	
	2	3	0	1	0	8	
	3	8	1	0	0	6	
21	1	2	0	5	0	6	
	2	3	3	0	1	0	
	3	9	2	0	0	1	
22	1	2	4	0	0	4	
	2	5	0	2	6	0	
	3	8	3	0	0	2	
23	1	3	5	0	1	0	
	2	4	4	0	1	0	
	3	7	4	0	0	5	
24	1	2	0	3	0	6	
	2	3	0	3	7	0	
	3	10	0	3	5	0	
25	1	2	0	5	0	9	
	2	3	4	0	5	0	
	3	9	0	1	0	6	
26	1	2	1	0	0	6	
	2	3	0	4	0	5	
	3	5	1	0	3	0	
27	1	3	3	0	8	0	
	2	7	3	0	7	0	
	3	10	0	2	0	2	
28	1	1	4	0	7	0	
	2	3	0	3	5	0	
	3	4	0	1	4	0	
29	1	6	0	3	2	0	
	2	9	5	0	1	0	
	3	10	5	0	0	4	
30	1	1	0	1	10	0	
	2	2	1	0	0	2	
	3	3	0	1	0	1	
31	1	4	5	0	1	0	
	2	8	0	4	0	6	
	3	10	0	3	0	4	
32	1	4	3	0	10	0	
	2	6	0	4	0	3	
	3	10	0	3	0	3	
33	1	1	3	0	0	7	
	2	3	3	0	2	0	
	3	6	3	0	1	0	
34	1	2	2	0	10	0	
	2	7	2	0	0	2	
	3	7	0	3	10	0	
35	1	1	5	0	5	0	
	2	3	3	0	0	5	
	3	5	0	2	0	4	
36	1	1	0	3	0	2	
	2	8	1	0	0	2	
	3	8	1	0	3	0	
37	1	2	0	2	10	0	
	2	5	4	0	5	0	
	3	5	0	1	3	0	
38	1	2	0	4	10	0	
	2	3	0	4	8	0	
	3	9	2	0	0	2	
39	1	3	4	0	8	0	
	2	3	0	3	0	6	
	3	6	0	2	0	2	
40	1	3	5	0	0	7	
	2	6	0	2	0	6	
	3	8	2	0	3	0	
41	1	4	4	0	0	7	
	2	8	2	0	2	0	
	3	9	0	2	0	5	
42	1	1	0	1	0	9	
	2	5	3	0	0	7	
	3	6	0	1	0	6	
43	1	5	0	2	0	6	
	2	6	3	0	9	0	
	3	8	0	1	0	4	
44	1	7	0	3	0	2	
	2	7	5	0	3	0	
	3	7	5	0	0	1	
45	1	3	0	4	5	0	
	2	4	0	4	4	0	
	3	6	2	0	5	0	
46	1	1	5	0	7	0	
	2	3	0	2	5	0	
	3	9	4	0	3	0	
47	1	5	0	5	0	8	
	2	6	0	4	5	0	
	3	10	1	0	5	0	
48	1	9	1	0	0	10	
	2	10	0	3	4	0	
	3	10	1	0	0	4	
49	1	2	4	0	9	0	
	2	6	0	5	5	0	
	3	9	0	5	4	0	
50	1	4	0	2	5	0	
	2	8	0	2	3	0	
	3	10	0	2	1	0	
51	1	3	4	0	7	0	
	2	10	2	0	0	3	
	3	10	0	2	5	0	
52	1	1	0	4	6	0	
	2	3	0	2	0	4	
	3	6	1	0	2	0	
53	1	5	0	2	6	0	
	2	7	0	2	0	7	
	3	7	4	0	2	0	
54	1	2	0	4	2	0	
	2	4	0	4	0	7	
	3	5	2	0	0	6	
55	1	2	2	0	8	0	
	2	8	0	2	0	5	
	3	9	1	0	1	0	
56	1	3	0	3	6	0	
	2	3	0	1	0	6	
	3	4	0	1	3	0	
57	1	2	5	0	0	7	
	2	2	0	3	6	0	
	3	8	2	0	0	3	
58	1	1	0	3	0	9	
	2	6	3	0	3	0	
	3	10	0	2	0	5	
59	1	6	4	0	0	4	
	2	10	4	0	5	0	
	3	10	0	3	0	4	
60	1	2	0	3	6	0	
	2	2	1	0	0	7	
	3	7	1	0	0	5	
61	1	7	0	3	8	0	
	2	9	0	2	0	3	
	3	10	5	0	0	2	
62	1	2	4	0	8	0	
	2	7	4	0	7	0	
	3	8	4	0	6	0	
63	1	1	5	0	0	8	
	2	8	0	3	0	6	
	3	9	0	3	0	5	
64	1	4	0	3	7	0	
	2	6	0	3	6	0	
	3	7	0	1	0	2	
65	1	2	4	0	0	8	
	2	3	0	3	4	0	
	3	8	4	0	3	0	
66	1	1	5	0	0	8	
	2	4	5	0	0	6	
	3	6	0	2	0	5	
67	1	2	4	0	4	0	
	2	9	4	0	3	0	
	3	10	4	0	2	0	
68	1	3	2	0	5	0	
	2	4	0	2	0	4	
	3	4	2	0	0	2	
69	1	5	4	0	0	1	
	2	6	0	4	7	0	
	3	10	4	0	6	0	
70	1	6	0	1	4	0	
	2	7	0	1	3	0	
	3	9	2	0	4	0	
71	1	3	0	3	0	8	
	2	7	1	0	0	8	
	3	10	1	0	3	0	
72	1	1	3	0	0	5	
	2	3	0	2	0	5	
	3	8	3	0	0	3	
73	1	3	2	0	10	0	
	2	4	2	0	6	0	
	3	8	0	1	0	3	
74	1	2	0	2	0	7	
	2	3	0	2	0	5	
	3	6	3	0	0	3	
75	1	6	4	0	0	7	
	2	8	0	3	0	5	
	3	10	0	3	0	2	
76	1	2	4	0	0	8	
	2	4	0	2	4	0	
	3	5	0	2	2	0	
77	1	1	3	0	0	5	
	2	3	0	3	0	5	
	3	8	0	3	2	0	
78	1	1	0	5	0	6	
	2	1	3	0	0	4	
	3	7	3	0	3	0	
79	1	4	3	0	0	8	
	2	5	0	1	5	0	
	3	8	3	0	0	4	
80	1	1	0	4	3	0	
	2	3	0	3	0	4	
	3	8	1	0	0	1	
81	1	3	2	0	7	0	
	2	4	2	0	0	4	
	3	8	0	2	5	0	
82	1	2	0	4	4	0	
	2	3	2	0	0	9	
	3	5	2	0	2	0	
83	1	3	0	2	0	6	
	2	4	2	0	4	0	
	3	6	2	0	1	0	
84	1	1	0	3	6	0	
	2	4	3	0	6	0	
	3	6	0	1	3	0	
85	1	4	2	0	0	4	
	2	5	0	4	5	0	
	3	6	2	0	4	0	
86	1	8	2	0	0	2	
	2	10	0	3	9	0	
	3	10	2	0	9	0	
87	1	4	0	5	0	6	
	2	4	3	0	4	0	
	3	10	1	0	3	0	
88	1	5	0	5	0	9	
	2	5	4	0	4	0	
	3	9	0	3	3	0	
89	1	4	4	0	0	3	
	2	6	0	3	0	2	
	3	7	0	3	0	1	
90	1	2	0	2	0	5	
	2	8	0	1	6	0	
	3	8	0	1	0	3	
91	1	1	0	5	0	8	
	2	3	3	0	0	3	
	3	6	0	3	0	2	
92	1	2	5	0	0	2	
	2	5	3	0	5	0	
	3	6	2	0	4	0	
93	1	1	0	4	0	4	
	2	9	3	0	4	0	
	3	10	2	0	0	4	
94	1	3	0	3	0	6	
	2	5	0	3	0	5	
	3	6	0	3	0	4	
95	1	5	0	3	0	5	
	2	5	3	0	4	0	
	3	10	0	2	0	4	
96	1	7	0	5	0	4	
	2	7	2	0	0	3	
	3	8	2	0	0	2	
97	1	4	0	3	4	0	
	2	5	0	3	0	2	
	3	8	0	3	2	0	
98	1	6	0	2	5	0	
	2	8	0	2	4	0	
	3	9	0	2	3	0	
99	1	1	0	4	8	0	
	2	2	0	4	5	0	
	3	10	0	3	4	0	
100	1	1	0	3	0	5	
	2	3	1	0	8	0	
	3	9	0	2	0	2	
101	1	3	0	5	6	0	
	2	6	0	4	0	5	
	3	10	3	0	1	0	
102	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	19	23	166	157

************************************************************************
