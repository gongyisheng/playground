use std::io::Result;

mod append;

#[tokio::main]
async fn main() -> Result<()> {
    append::fs_append_test().await?;
    Ok(())
}
