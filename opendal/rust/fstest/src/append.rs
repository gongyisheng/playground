//use std::sync::Arc;

use opendal::services::Fs;
use opendal::Operator;
use std::io::Result;

pub async fn append_test() -> Result<()> {
    // Create fs backend builder.
    let mut builder = Fs::default();
    // Set the root for fs, all operations will happen under this root.
    //
    // NOTE: the root must be absolute path.
    builder.root(".");

    // `Accessor` provides the low level APIs, we will use `Operator` normally.
    let op: Operator = Operator::new(builder)?.finish();

    op.write_with("test.txt", "hello world!\n")
        .append(true)
        .await?;
    Ok(())
}
