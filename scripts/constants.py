STACK_MAX = 30 # 选用20作为stack归一化的基底 50 -> 30
SUCCESSOR_MAX = 2 # 这个不能称之为选用,因为2就是确切最大的后继节点数
TEST_CASE_NUMBER_MAX = 50 # 选用50作为test_case_number归一化的基底
# !!! branch_new_instruction和path_new_instruction有待考虑重新选用基底,而不是使用当前字节码长度作为基底,因为当前的百分比很容易极小 !!!
DEPTH_MAX = 15 # 选用20作为depth归一化的基底 20 -> 15
# cpicnt和covNew使用当前字节码长度作为基底
ICNT_MAX = 10 # 选用10作为icnt归一化的基底
SUBPATH_MAX = 5 # 选用5作为subpath归一化的基底 10 -> 5
REWARD_MAX = 100  # 选用150作为reward归一化的基底 150 -> 100

