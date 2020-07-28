import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import CircularProgress from '@material-ui/core/CircularProgress';
import Fade from '@material-ui/core/Fade';
import Box from '@material-ui/core/Box';
import Typography from '@material-ui/core/Typography';
import Cookies from 'universal-cookie';
import { BASEURL } from '../Constants';


const useStyles = makeStyles((theme) => ({
  dialogInputBox: {
    margin: theme.spacing(0, 0, 2),
  },
  dialogActions: {
    display: "flex",
    margin: theme.spacing(0, 2, 2),
  },
  errorMessage: {
    color: "#f44336",
    margin: theme.spacing(0, 2, 0),
  },
  circleProgress: {
    zIndex: 999,
    position: "absolute",
    top: "50%",
    left: "50%",
    marginTop: "-24px",
    marginLeft: "-24px"
  },
}));

export default function CreatePortfolioDialog(props) {

  const classes = useStyles();

  const [loading, setLoading] = React.useState(false);
  const [errorMsgIsShow, setErrorMsgShow] = React.useState(false);
  const [nameInputIsError, setNameError] = React.useState(false);

  const [nameInputErrorMsg, setNameErrorMsg] = React.useState("");
  const [errorMsg, setErrorMsg] = React.useState("");

  const nameInput = React.useRef();


  const checkNameInputEmpty = () => {
    if (nameInput.current.value.length < 1) {
      setNameError(true);
      setNameErrorMsg("Portfolio's name cannot be empty");
    } else {
      setNameError(false);
      setNameErrorMsg("");
    }
  }

  // Do validate here when the focus of input field is out
  // Check whether the input is empty
  const nameInputOnBlur = () => {
    checkNameInputEmpty();
  };

  const createButtonOnClick = async (e) => {
    // If no error in textfield
    checkNameInputEmpty();
    setErrorMsg("");
    if (nameInput.current.value.length > 0) {
      const request = {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          'portfolioName': nameInput.current.value,
          'userId': props.userData.userId,
        })
      }

      try {
        setLoading(true);
        const response = await fetch(BASEURL + "/portfolio/create", request)
        if (response.ok) {
          const jsonData = await response.json();
          if (jsonData.isSuccess) {
            // get create object
            newPortfolio = {
              'portfolioId': jsonData.id,
              'user_id': jsonData.user_id,
              'portfolio_name': jsonData.portfolio_name,
              'portfolio_stocks': jsonData.portfolio_stocks
            }
            setCurrentSelectedPortfolio(jsonData.id);
            setUserPortfolios([...userPortfolios, newPortfolio]);
            props.handleClose();
          } else {
            setErrorMsg(jsonData.errorMsg);
          }
        }
      }
      catch (err) {
        alert('create new portfolio failed', err);
      }
      finally {
        setLoading(false);
      }
    }
  };

  return (
    <div>
      <Dialog open={props.isOpen} onClose={props.handleClose} aria-labelledby="form-dialog-title">
        <Fade
          in={loading}
          unmountOnExit
        >
          <CircularProgress className={classes.circleProgress} />
        </Fade>
        <DialogTitle id="form-dialog-title">Create New Portfolio</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            inputRef={nameInput}
            margin="dense"
            id="portfolioName"
            label="Portfolio name"
            type="text"
            variant="outlined"
            fullWidth
            error={nameInputIsError}
            helperText={nameInputErrorMsg}
            className={classes.dialogInputBox}
            onBlur={nameInputOnBlur}
          />
          <Typography className={classes.errorMessage}>
            {errorMsg}
          </Typography>
        </DialogContent>
        <DialogActions className={classes.dialogActions}>
          <Button variant="contained" onClick={createButtonOnClick} color="primary">
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}