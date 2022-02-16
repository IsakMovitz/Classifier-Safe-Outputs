import rubrix as rb

''' Follow github 
python -m rubrix
'''

### Write to rubrix ###

rb.log(
  rb.TextClassificationRecord(inputs="hejhej"), 
  name='new_testing'
)

### Read from rubrix ###

rb_df = rb.load(name='new_testing')       # Pandas dataframe , query="status:Validated", query="status:Default"

print(rb_df['inputs'])