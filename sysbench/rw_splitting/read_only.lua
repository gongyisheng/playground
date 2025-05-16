-- custom_query.lua

function thread_init()
  -- Connect once per thread
  drv = sysbench.sql.driver()
  con = drv:connect()
end
  
function event()
  -- Your custom query goes here
  con:query("SELECT * FROM test WHERE id = " .. math.random(1, 1000000))
end