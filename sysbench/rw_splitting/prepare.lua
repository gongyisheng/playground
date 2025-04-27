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
    local n1 = sysbench.rand.default(1, 4294967295)
    local n2 = sysbench.rand.default(1, 4294967295)
    local n3 = sysbench.rand.default(1, 4294967295)

    local p1 = random_string(256)
    local p2 = random_string(256)
    local p3 = random_string(256)

    local query = string.format(
        "INSERT INTO test (number1, number2, number3, payload1, payload2, payload3) VALUES (%d, %d, %d, %s, %s, %s)",
        n1, n2, n3, p1, p2, p3
    )
    con:query(query)
end