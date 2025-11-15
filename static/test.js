async function get_db_posts(id){    
    const res = await fetch("/db/uuid?" + new URLSearchParams({uuid: id}).toString());
    const data = await res.json();

    document.getElementById("resp").textContent = JSON.stringify(data) || data.error;
  }

  // Just have few pictures for the Recent Pictures