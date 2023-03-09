- whenMatchedUpdateAll():  
Update all the columns of the matched table row with the values of the corresponding columns in the source row. If a condition is specified, then it must be true for the new row to be updated.
- whenNotMatchedInsertAll():  
Insert all the columns of the source row into the target table. If a condition is specified, then it must be true for the new row to be inserted.

- demo code:  
```
base.alias('base').merge(updates.alias('updates'), 'base.<key> == updates.<key>').whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
```