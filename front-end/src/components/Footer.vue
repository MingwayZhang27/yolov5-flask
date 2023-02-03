<template>
  <div id="Footer">
    <p>{{ msg }}</p>
  </div>
</template>
<script>
export default {
  name: "Footer",
  data() {
    return {
      msg: "",
    };
  },
};
</script>
<style scoped>
#Footer {
  /*background:#F2F6FC;*/
  padding: 6px;
  border-radius: 5px;
  width: 80%;
  height: 80px;
  margin: 20px auto;
  margin-top: 140px;
}

p {
  color: #2142b9d1;
  margin: 30px auto;
  font-size: 1.1em;
}
</style>


