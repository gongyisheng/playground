use std::io::Result;

mod append;

#[tokio::main]
async fn main() -> Result<()> {
    append::append_test().await?;
    Ok(())
}
