// console.log("hi");
class chatbot {
  constructor() {
    this.assistant_name = "Plato";
    this.assistant_description = "Your personal asssistant";
    this.input_placeholder = "Write your query...";
    this.reply_time = 1; // in seconds

    // elements are here
    this.assistant_name_ele = document.getElementById("assistant_name");
    this.assistant_description_ele = document.getElementById(
      "assistant_description"
    );
    this.options_ele = document.getElementById("options");
    this.chat_ele = document.getElementById("chat");
    this.input_ele = document.getElementById("input");
    this.chatbot_ele = document.getElementById("chatbot");
    this.chat_circle_ele = document.getElementById("chat_circle");
    this.cross = document.getElementById("cross");
    this.msg_audio = document.getElementById("msg_audio");
    this.msg_audio_s = document.getElementById("msg_audio_s");
    this.send_ele = document.getElementById("send");

    this.options = Object.keys(options);
    this.questions = Object.keys(qa);
    this.answers = Object.values(qa);

    for (let i = 0; i < Object.keys(options).length; i++) {
      this.questions.push(Object.keys(options)[i]);
      this.answers.push(Object.values(options)[i]);
    }

    console.log(this.questions);
    console.log(this.answers);
    this.setThings();
  }

  setThings = () => {
    this.assistant_name_ele.innerHTML += this.assistant_name;
    this.assistant_description_ele.innerText = this.assistant_description;
    for (let i = 0; i < this.options.length; i++) {
      this.options_ele.innerHTML +=
        `<div onclick='cb.option_send(this)'>` + this.options[i] + `</div>`;
    }
    this.input_ele.placeholder = this.input_placeholder;

    //putting the functions together
    this.chat_circle_ele.onclick = this.open;
    this.cross.onclick = this.close;
    this.send_ele.onclick = this.send;

    this.input_ele.onkeydown = (event) => {
      if (event.key === "Enter") {
        this.send();
      }
    };
  };

  sendMessage(message) {
    fetch("/get", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: `msg=${message}`,
    })
      .then((response) => response.text())
      .then((data) => {
        console.log(data)
        if (data !== "Please give me more info") {
          this.chat_ele.innerHTML += this.return_chatbot_msg(data);
        } else {
          document.getElementById("input").value = "";

          this.get(message);
        }
      })
      .catch((error) => {
        console.log(error);
      });
  }

  close = () => {
    this.chatbot_ele.style.display = "none";
    this.chat_circle_ele.style.display = "flex";
  };

  open = () => {
    this.chat_circle_ele.style.display = "none";
    this.chatbot_ele.style.display = "flex";
  };

  return_user_msg = (msg) => {
    return (
      `<div class="full_right">
<div class="small_right">
<div>` +
      msg +
      `</div>
</div>
</div>`
    );
  };

  return_chatbot_msg = (msg) => {
    return (
      `<div class="full">
      <img src="../static/bot.png" alt="" class="bot_img" />
<div class="small">
<div>` +
      msg +
      `</div>
</div>
</div>`
    );
  };

  scroll_up = () => {
    this.chat_ele.scrollTo(0, 500);
  };

  send = () => {
    // this.msg_audio_s.play();
    this.chat_ele.innerHTML += this.return_user_msg(this.input_ele.value);
    //    this.get(this.input_ele.value);
    this.sendMessage(this.input_ele.value);
    this.input_ele.value = "";
    this.scroll_up();
  };

  get = (msg) => {
    msg = msg.toLowerCase();

    setTimeout(() => {
      // this.msg_audio.play();

      this.link = false;
      // getting the links first
      for (let i = 0; i < Object.keys(links).length; i++) {
        if (msg.includes(Object.keys(links)[i])) {
          this.link = true;
          for (let j = 0; j < Object.values(links)[i].length; j++) {
            this.chat_ele.innerHTML += this.return_chatbot_msg(
              "" +
                Object.values(links)[i][j].title +
                "<br>" +
                Object.values(links)[i][j].content +
                "<br>" +
                "If you want to read more the click here <a href='" +
                Object.values(links)[i][j].link +
                "'>link</a>"
            );
          }
        }
      }

      if (this.link == false) {
        this.confidence = [];

        let user = msg;
        for (let i = 0; i < this.questions.length; i++) {
          this.confidence.push(0);
          this.questions[i] = this.questions[i].toLowerCase();
          let indi_questions_split = this.questions[i].split(" ");
          let user_split = user.split(" ");
          // console.log(msg);
          for (let j = 0; j <= user_split.length; j++) {
            if (indi_questions_split.includes(user_split[j])) {
              this.confidence[i] += 1;
            }
          }
        }

        // console.log(this.confidence);
        let percent_array = [];

        for (let i = 0; i < this.confidence.length; i++) {
          let user_split = user.split(" ");
          let percent = parseInt(
            (this.confidence[i] / user_split.length) * 100
          );
          percent_array.push(percent);
        }

        // console.log(percent_array);

        let max = -1;
        for (let i = 0; i < percent_array.length; i++) {
          if (percent_array[i] > max && percent_array[i] > 30) {
            max = i;
          }
        }

        // console.log(max);
        if (max == -1) {
          this.chat_ele.innerHTML += this.return_chatbot_msg(
            "I am not able to understand you."
          );
        } else {
          this.chat_ele.innerHTML += this.return_chatbot_msg(this.answers[max]);
        }

        this.scroll_up();
      }
    }, 1500);
  };

  option_send = (ele) => {
    // console.log(ele.innerText);
    this.msg_audio_s.play();
    // this.chat_ele.innerHTML += this.return_user_msg(ele.innerText);
    this.get(ele.innerText);
    this.input_ele.value = "";
    this.scroll_up();
  };
}

console.log("hello");

function dpop() {
  document.getElementsByClassName("popup")[0].style.display = "none";
}

let cb = new chatbot();
