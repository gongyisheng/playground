mysql 8.0.38, aws r7g.x2large
| Method                | Threads | Read QPS           | Write QPS          | Transaction QPS    |
|-----------------------|---------|--------------------|--------------------|--------------------|
| oltp_insert           | 1       | 0.0                | 293.9177030431479  | 293.9177030431479  |
| oltp_insert           | 2       | 0.0                | 562.4781413939545  | 562.4781413939545  |
| oltp_insert           | 4       | 0.0                | 867.3316409139065  | 867.3316409139065  |
| oltp_insert           | 8       | 0.0                | 1554.878048780488  | 1554.878048780488  |
| oltp_insert           | 16      | 0.0                | 2891.454272863568  | 2891.454272863568  |
| oltp_insert           | 32      | 0.0                | 5427.794930207927  | 5427.794930207927  |
| oltp_insert           | 64      | 0.0                | 9531.404488210208  | 9531.404488210208  |
| oltp_point_select     | 1       | 10463.38146474141  | 0.0                | 10463.38146474141  |
| oltp_point_select     | 2       | 20897.855107244635 | 0.0                | 20897.855107244635 |
| oltp_point_select     | 4       | 29933.304668673194 | 0.0                | 29933.304668673194 |
| oltp_point_select     | 8       | 33207.483027206465 | 0.0                | 33207.483027206465 |
| oltp_point_select     | 16      | 41606.84649377137  | 0.0                | 41606.84649377137  |
| oltp_point_select     | 32      | 54091.263494602164 | 0.0                | 54091.263494602164 |
| oltp_point_select     | 64      | 61542.681281787394 | 0.0                | 61542.681281787394 |
| select_random_points  | 1       | 6750.162491875406  | 0.0                | 6750.162491875406  |
| select_random_points  | 2       | 12260.186990650467 | 0.0                | 12260.186990650467 |
| select_random_points  | 4       | 21406.073453389195 | 0.0                | 21406.073453389195 |
| select_random_points  | 8       | 24228.781107211777 | 0.0                | 24228.781107211777 |
| select_random_points  | 16      | 29092.072378286513 | 0.0                | 29092.072378286513 |
| select_random_points  | 32      | 35931.496991624525 | 0.0                | 35931.496991624525 |
| select_random_points  | 64      | 21208.636399968025 | 0.0                | 21208.636399968025 |
| select_random_ranges  | 1       | 6236.98815059247   | 0.0                | 6236.98815059247   |
| select_random_ranges  | 2       | 12032.2780633162   | 0.0                | 12032.2780633162   |
| select_random_ranges  | 4       | 21535.200423944887 | 0.0                | 21535.200423944887 |
| select_random_ranges  | 8       | 24973.40425531915  | 0.0                | 24973.40425531915  |
| select_random_ranges  | 16      | 29701.48656889502  | 0.0                | 29701.48656889502  |
| select_random_ranges  | 32      | 36508.455265051576 | 0.0                | 36508.455265051576 |
| select_random_ranges  | 64      | 28554.960642506096 | 0.0                | 28554.960642506096 |
| oltp_read_only        | 1       | 7233.4383280835955 | 0.0                | 516.6741662916854  |
| oltp_read_only        | 2       | 13923.989881315429 | 0.0                | 994.5707058082448  |
| oltp_read_only        | 4       | 22282.874938764086 | 0.0                | 1591.6339241974347 |
| oltp_read_only        | 8       | 24866.156214389954 | 0.0                | 1776.1540153135682 |
| oltp_read_only        | 16      | 29919.56354479961  | 0.0                | 2137.1116817714005 |
| oltp_read_only        | 32      | 38276.65078556517  | 0.0                | 2734.046484683227  |
| oltp_read_only        | 64      | 43328.44267449172  | 0.0                | 3094.8887624636945 |
| oltp_read_write       | 1       | 2161.91904047976   | 617.6911544227886  | 154.42278860569715 |
| oltp_read_write       | 2       | 4207.738208277944  | 1202.2109166508412 | 300.5527291627103  |
| oltp_read_write       | 4       | 6863.321142354657  | 1960.9488978156164 | 490.2372244539041  |
| oltp_read_write       | 8       | 11272.080924278282 | 3220.0952637728046 | 804.9489230400527  |
| oltp_read_write       | 16      | 16098.855068416644 | 4596.281964335581  | 1148.6216938604541 |
| oltp_read_write       | 32      | 17432.63995532598  | 4966.993079515765  | 1239.8037534153686 |
| oltp_read_write       | 64      | 15464.502559625384 | 4367.732846541529  | 1085.4597404658916 |
| oltp_write_only       | 1       | 0.0                | 834.5913521619595  | 208.6478380404899  |
| oltp_write_only       | 2       | 0.0                | 1717.3645751072104 | 429.3411437768026  |
| oltp_write_only       | 4       | 0.0                | 2580.7838002618664 | 645.1709628089674  |
| oltp_write_only       | 8       | 0.0                | 4166.383667272618  | 1041.271299866158  |
| oltp_write_only       | 16      | 0.0                | 5671.817601022201  | 1417.1058936272161 |
| oltp_write_only       | 32      | 0.0                | 6340.267325844488  | 1581.7277502566608 |
| oltp_write_only       | 64      | 0.0                | 4851.792986987186  | 1203.7349756630574 |
| oltp_update_index     | 1       | 0.0                | 239.56885604871366 | 239.56885604871366 |
| oltp_update_index     | 2       | 0.0                | 495.2177258962392  | 495.2177258962392  |
| oltp_update_index     | 4       | 0.0                | 803.6383627367684  | 803.6383627367684  |
| oltp_update_index     | 8       | 0.0                | 1380.8049535603716 | 1380.8049535603716 |
| oltp_update_index     | 16      | 0.0                | 2349.0616943461837 | 2349.0616943461837 |
| oltp_update_index     | 32      | 0.0                | 3594.705294705295  | 3594.705294705295  |
| oltp_update_index     | 64      | 0.0                | 4328.022998143379  | 4328.022998143379  |
| oltp_update_non_index | 1       | 0.0                | 274.0643716316879  | 274.0643716316879  |
| oltp_update_non_index | 2       | 0.0                | 538.5761274906771  | 538.5761274906771  |
| oltp_update_non_index | 4       | 0.0                | 830.1758525198196  | 830.1758525198196  |
| oltp_update_non_index | 8       | 0.0                | 1524.9612713007846 | 1524.9612713007846 |
| oltp_update_non_index | 16      | 0.0                | 2782.012728162809  | 2782.012728162809  |
| oltp_update_non_index | 32      | 0.0                | 4619.781933020858  | 4619.781933020858  |
| oltp_update_non_index | 64      | 0.0                | 7504.467182430746  | 7504.467182430746  |

pgsql 17.2, aws r7g.x2large
| Method                | Threads | Read QPS           | Write QPS          | Transaction QPS    |
|-----------------------|---------|--------------------|--------------------|--------------------|
| oltp_insert           | 1       | 0.0                | 1103.4565506484157 | 1103.4565506484157 |
| oltp_insert           | 2       | 0.0                | 1178.011518157095  | 1178.011518157095  |
| oltp_insert           | 4       | 0.0                | 2532.9200619907015 | 2532.9200619907015 |
| oltp_insert           | 8       | 0.0                | 4942.06332543515   | 4942.06332543515   |
| oltp_insert           | 16      | 0.0                | 9374.625194898652  | 9374.625194898652  |
| oltp_insert           | 32      | 0.0                | 17738.348568859587 | 17738.348568859587 |
| oltp_insert           | 64      | 0.0                | 29120.886795841398 | 29120.886795841398 |
| oltp_point_select     | 1       | 18270.35188944332  | 0.0                | 18270.35188944332  |
| oltp_point_select     | 2       | 33201.17195312187  | 0.0                | 33201.17195312187  |
| oltp_point_select     | 4       | 45209.687418754875 | 0.0                | 45209.687418754875 |
| oltp_point_select     | 8       | 49688.937327520696 | 0.0                | 49688.937327520696 |
| oltp_point_select     | 16      | 67496.02583457474  | 0.0                | 67496.02583457474  |
| oltp_point_select     | 32      | 78265.51713803335  | 0.0                | 78265.51713803335  |
| oltp_point_select     | 64      | 84320.80535573537  | 0.0                | 84320.80535573537  |
| select_random_points  | 1       | 17486.70053197872  | 0.0                | 17486.70053197872  |
| select_random_points  | 2       | 29363.125474981    | 0.0                | 29363.125474981    |
| select_random_points  | 4       | 28926.28589712823  | 0.0                | 28926.28589712823  |
| select_random_points  | 8       | 44840.425531914894 | 0.0                | 44840.425531914894 |
| select_random_points  | 16      | 61207.63770868739  | 0.0                | 61207.63770868739  |
| select_random_points  | 32      | 47359.50062421973  | 0.0                | 47359.50062421973  |
| select_random_points  | 64      | 77548.27892493937  | 0.0                | 77548.27892493937  |
| select_random_ranges  | 1       | 188.50763126068227 | 0.0                | 188.50763126068227 |
| select_random_ranges  | 2       | 4924.255302128851  | 0.0                | 4924.255302128851  |
| select_random_ranges  | 4       | 699.9850067469639  | 0.0                | 699.9850067469639  |
| select_random_ranges  | 8       | 2879.0831060083738 | 0.0                | 2879.0831060083738 |
| select_random_ranges  | 16      | 14428.771368589423 | 0.0                | 14428.771368589423 |
| select_random_ranges  | 32      | 6726.913442249923  | 0.0                | 6726.913442249923  |
| select_random_ranges  | 64      | 6048.680369436779  | 0.0                | 6048.680369436779  |
| oltp_read_only        | 1       | 11116.066072071351 | 0.0                | 794.0047194336679  |
| oltp_read_only        | 2       | 26323.92493276143  | 0.0                | 1880.2803523401024 |
| oltp_read_only        | 4       | 33549.8320268757   | 0.0                | 2396.416573348264  |
| oltp_read_only        | 8       | 33596.138857245634 | 0.0                | 2399.724204088974  |
| oltp_read_only        | 16      | 45322.09079283887  | 0.0                | 3237.292199488491  |
| oltp_read_only        | 32      | 56112.65861962241  | 0.0                | 4008.0470442587434 |
| oltp_read_only        | 64      | 53220.21055989791  | 0.0                | 3801.4436114212795 |
| oltp_read_write       | 1       | 6331.440227172739  | 1808.9829220493539 | 452.24573051233847 |
| oltp_read_write       | 2       | 11902.981344104297 | 3238.6874887524746 | 796.6247425566376  |
| oltp_read_write       | 4       | 17525.367136187782 | 4407.03381951595   | 1055.1728964021154 |
| oltp_read_write       | 8       | 18934.163293362868 | 4248.548842893149  | 969.6020774459636  |
| oltp_read_write       | 16      | 20302.744350296227 | 3820.493613378343  | 810.3664646425341  |
| oltp_read_write       | 32      | 19931.389365351628 | 3069.852070164798  | 595.4152175355135  |
| oltp_write_only       | 1       | 0.0                | 3254.841967383588  | 813.710491845897   |
| oltp_write_only       | 2       | 0.0                | 3852.2836346184613 | 947.8483442649176  |
| oltp_write_only       | 4       | 0.0                | 3667.04635736233   | 879.8613969426922  |
| oltp_write_only       | 8       | 0.0                | 2981.1159988903423 | 685.3128839218484  |
| oltp_write_only       | 16      | 0.0                | 3200.0401156058824 | 687.70912538862    |
| oltp_write_only       | 32      | 0.0                | 1546.9272594779425 | 300.0803452979316  |
| oltp_write_only       | 64      | 0.0                | 577.9749427192512  | 100.13162384829133 |
| oltp_update_index     | 1       | 0.0                | 1110.3889611038896 | 1110.3889611038896 |
| oltp_update_index     | 2       | 0.0                | 1232.2521297444305 | 1232.2521297444305 |
| oltp_update_index     | 4       | 0.0                | 2272.9045224120864 | 2272.9045224120864 |
| oltp_update_index     | 8       | 0.0                | 4427.747282961896  | 4427.747282961896  |
| oltp_update_index     | 16      | 0.0                | 9274.044466650012  | 9274.044466650012  |
| oltp_update_index     | 32      | 0.0                | 13086.705432928715 | 13086.705432928715 |
| oltp_update_index     | 64      | 0.0                | 20674.984292253837 | 20674.984292253837 |
| oltp_update_non_index | 1       | 0.0                | 1191.840407979601  | 1191.840407979601  |
| oltp_update_non_index | 2       | 0.0                | 1304.3304370431842 | 1304.3304370431842 |
| oltp_update_non_index | 4       | 0.0                | 2545.967185578452  | 2545.967185578452  |
| oltp_update_non_index | 8       | 0.0                | 4755.743966086105  | 4755.743966086105  |
| oltp_update_non_index | 16      | 0.0                | 8724.87179743495   | 8724.87179743495   |
| oltp_update_non_index | 32      | 0.0                | 15561.049182947781 | 15561.049182947781 |
| oltp_update_non_index | 64      | 0.0                | 21503.456393579985 | 21503.456393579985 |