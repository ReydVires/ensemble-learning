-- Ahmad Arsyel 1301164193
--- Prepare data
local TRAIN_PATH = "D:/Telkom University/Machine Learning/Tupro4/TrainsetTugas4ML.xls"
local TEST_PATH = "D:/Telkom University/Machine Learning/Tupro4/TestsetTugas4ML.xls"
local ANSWER_PATH = "D:/Telkom University/Machine Learning/Tupro4/TebakanTugas4ML.csv"

-- @param table Print traverse of table
local function print_table(table)
  for _, v in ipairs(table) do -- show parsing from CSV file
    print(v.id, v.x, v.y, v.label)
  end
end

-- @param path Use your path for .csv file
-- @return Table that contain parsed from CSV file (testset)
local function parse_CSV(path)
  local tab_list = {}
  print("load CSV file to table (parse_CSV)")
  -- output will saved in table_list
  for line in io.lines(path) do
    local col1, col2, col3 = line:match("%s*(.*),%s*(.*),%s*(.*)") -- converting
    tab_list[#tab_list + 1] = {
      id = #tab_list,
      x = tonumber(col1),
      y = tonumber(col2),
      label = tonumber(col3)
    }
  end
  table.remove(tab_list, 1) -- remove the title/header
  return tab_list
end

local function parse_testCSV(path)
  local tab_list = {}
  print("load CSV file to table (parse_testCSV)")
  -- output will saved in table_list
  for line in io.lines(path) do
    local col1, col2, col3 = line:match(
        "%s*(.*),%s*(.*),%s*(.*)") -- converting
    tab_list[#tab_list + 1] = {
      id = #tab_list,
      x = tonumber(col1),
      y = tonumber(col2),
      label = '?'
    }
  end
  table.remove(tab_list, 1) -- remove the title/header
  return tab_list
end

-- @param path Use your own path for .csv raw file targeted
-- @param data_table Saving file to .csv from data_table
-- @param sep Separator of file
local function table_to_CSV(path, data_table, sep)
  sep = sep or ','
  local file = assert(io.open(path, "w")) -- w mean write
  for _, v in ipairs(data_table) do
    file:write(v.label) -- v.y can be replaced (the solution column)
    file:write('\n')
  end
  file:close()
  print("\nthe result is saved to CSV file\n")
end

local function counting_label(tab, col_name, label)
  local sum = 0
  for _, v in pairs(tab) do
    if (v[col_name] == label) then
      sum = sum + 1
    end
  end
  return sum
end

local function sum_attrb(tab, col_name, label)
  local sum = 0
  for _, v in pairs(tab) do
    if (v.label == label) then
      sum = sum + v[col_name]
    end
  end
  return sum
end

local function std(tab, col_name, avg, n, label)
  local result = 0
  local dt
  for _, v in pairs(tab) do
    if (v.label == label) then
      dt = v[col_name] - avg
      result = result + dt^2
    end
  end
  return math.sqrt(result / (n-1))
end

local function p_count(stdev, curr_val, avg)
  return 1 * math.exp(-(curr_val - avg)^2 / (2 * (stdev)^2)) / 
    (stdev * math.sqrt(2 * math.pi))
end

local function create_bootstraps(tab, n)
  local bootstraps = {}
  for i=1, n do
    bootstraps[i] = {}
    for j=1, #tab do
      bootstraps[i][j] = {}
      for k, v in pairs(tab[math.random(1, #tab)]) do
        bootstraps[i][j][k] = v
      end
    end
    table.sort(bootstraps[i], function(f_val, s_val) return f_val.id < s_val.id end)
    --print_table(bootstraps[i])
  end
  return bootstraps
end

local function create_test_template(n)
  local test_sets = {}
  for i=1, n do
    test_sets[i] = parse_testCSV(TEST_PATH)
  end
  return test_sets
end

local function create_train_template(n)
  local train_sets = {}
  for i=1, n do
    train_sets[i] = parse_CSV(TRAIN_PATH)
  end
  return train_sets
end

local function naive_bayes(tab_train, tab_test)
  local total_c1 = counting_label(tab_train, 'label', 1)
  local total_c2 = counting_label(tab_train, 'label', 2)
  local avg_c1 = total_c1/#tab_train
  local avg_c2 = total_c2/#tab_train

  local sum_x_c1 = sum_attrb(tab_train, 'x', 1)
  local sum_y_c1 = sum_attrb(tab_train, 'y', 1)
  local sum_x_c2 = sum_attrb(tab_train, 'x', 2)
  local sum_y_c2 = sum_attrb(tab_train, 'y', 2)

  local avg_x_c1 = sum_x_c1/total_c1
  local avg_y_c1 = sum_y_c1/total_c1
  local avg_x_c2 = sum_x_c2/total_c2
  local avg_y_c2 = sum_y_c2/total_c2

  local stdx_c1 = std(tab_train, 'x', avg_x_c1, total_c1, 1)
  local stdy_c1 = std(tab_train, 'y', avg_y_c1, total_c1, 1)
  local stdx_c2 = std(tab_train, 'x', avg_x_c2, total_c2, 2)
  local stdy_c2 = std(tab_train, 'y', avg_y_c2, total_c2, 2)
  
  --[
  print('Total Label 1: ' .. total_c1)
  print('Total Label 2: ' .. total_c2)
  print(string.format('AVG c1: %s | AVG c2: %s', avg_c1, avg_c2))

  print('AVG atribut x c1: ' .. avg_x_c1)
  print('AVG atribut y c1: ' .. avg_y_c1)

  print('AVG atribut x c2: ' .. avg_x_c2)
  print('AVG atribut y c2: ' .. avg_y_c2)

  print('STD x c1: ', stdx_c1)
  print('STD y c1: ', stdy_c1)
  print('STD x c2: ', stdx_c2)
  print('STD y c2: ', stdy_c2)
  
  
  for i = 1, #tab_test do
    local p1 = avg_c1
      * p_count(stdx_c1, tab_test[i].x, avg_x_c1)
      * p_count(stdy_c1, tab_test[i].y, avg_y_c1)
      
    local p2 = avg_c2
      * p_count(stdx_c2, tab_test[i].x, avg_x_c2)
      * p_count(stdy_c2, tab_test[i].y, avg_y_c2)
    
    p1 = 1/-(math.log(p1))
    p2 = 1/-(math.log(p2))
    --print(string.format('%s) Hasil label 1: %s | Hasil label 2: %s', i, p1, p2))
    if p1 > p2 then
      tab_test[i].label = 1
    else
      tab_test[i].label = 2
    end
  end
  return tab_test
end

local function accuracy(trainset1, trainset2)
  local count = 0
  for i=1, #trainset1 do
    if (trainset1[i].label == trainset2[i].label) then
      count = 1 + count
    end
  end
  print('ACCURACY: ', (count/#trainset1 * 100) .. ' %')
end

--- MAIN PROGRAM
math.randomseed(os.time() * 2)
local trainset = parse_CSV(TRAIN_PATH)
--print_table(trainset)
print(string.format('Total trainset: %s', #trainset))
local testset = parse_testCSV(TEST_PATH)
print('Total testset:', #testset)

--> BAGGIN
local nmodel = 21 -- optimal accuracy in ~94.630872483221 %
local models = create_bootstraps(trainset, nmodel)
local testsets = create_test_template(#models)
local trainset_test = create_train_template(#models)
local trainset_result = parse_CSV(TRAIN_PATH)
local count_most_label = {0, 0}

for i=1, nmodel do
  print(string.format('\nRESULT MODEL %s ------DETAIL BELOW--------------', i))
  naive_bayes(models[i], testsets[i])
  --print_table(trainset_test[i])
  naive_bayes(models[i], trainset_test[i])
end

for i=1, #testset do -- for i=1, #trainset_result do
  for j=1, nmodel do
    --count_most_label[trainset_test[j][i].label] = count_most_label[trainset_test[j][i].label] + 1
    count_most_label[testsets[j][i].label] = count_most_label[testsets[j][i].label] + 1
  end
  --print(string.format('Data %s> vote (c1: %s, c2: %s)', i, count_most_label[1], count_most_label[2]))
  if count_most_label[1] > count_most_label[2] then
    testset[i].label = 1
  else
    testset[i].label = 2
  end
  count_most_label = {0, 0}
end

--- ACCURACY
for i=1, #trainset_result do
  for j=1, nmodel do
    count_most_label[trainset_test[j][i].label] = count_most_label[trainset_test[j][i].label] + 1
  end
  if count_most_label[1] > count_most_label[2] then
    trainset_result[i].label = 1
  else
    trainset_result[i].label = 2
  end
  count_most_label = {0, 0}
end

--> RESULT
print('\n-------> Last result:')
print_table(testset)
table_to_CSV(ANSWER_PATH, testset)

accuracy(trainset, trainset_result)