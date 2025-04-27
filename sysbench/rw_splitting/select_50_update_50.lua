function thread_init()
  drv = sysbench.sql.driver()
  con = drv:connect()
end

function random_string(length)
  local res = ""
  for i = 1, length do
    res = res .. string.char(sysbench.rand.default(32, 126))
  end
  return "'" .. res .. "'"
end

function event()
  local id = math.random(1, 1000000)
  if id % 2 == 0 then
    -- Write
    local n1 = sysbench.rand.default(1, 4294967295)
    local n2 = sysbench.rand.default(1, 4294967295)
    local n3 = sysbench.rand.default(1, 4294967295)

    local p1 = random_string(256)
    local p2 = random_string(256)
    local p3 = random_string(256)
    local query = string.format(
        "UPDATE test SET number1=%d, number2=%d, number3=%d, payload1='%s', payload2='%s', payload3='%s' WHERE id=%d",
        n1, n2, n3, p1, p2, p3, id
    )
    con:query(query)
  else
    -- Read
    query = string.format("SELECT * FROM test WHERE id=%d", id)
    con:query(query)
  end
end